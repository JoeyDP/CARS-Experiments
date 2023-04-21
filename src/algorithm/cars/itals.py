from typing import List
import itertools
import warnings

import numpy as np
import scipy.sparse
from tqdm.auto import tqdm
import joblib
from joblib import Parallel, delayed

from .cars_algorithm import CARSAlgorithm
from src.data.cars import CARSData, CARSDataMD, CARSDataFlat, stackForContexts
from src.util import multiply, indexFactors, unstackAverageContexts
from src.algorithm import alg_util


class iTALS(CARSAlgorithm):
    """
    iTALS algorithm based on paper:
    Hidasi, Balázs, and Domonkos Tikk. "Fast ALS-based tensor factorization for context-aware recommendation
    from implicit feedback." Joint European Conference on Machine Learning and Knowledge Discovery in Databases.
    Springer, Berlin, Heidelberg, 2012.
    """
    Ms: List[np.array]
    l2s: List[np.array]

    def __init__(self, l2: float = 1, v=0, k=20, alpha: float = 1, max_iterations=10,
                 default_context_1=False,
                 n_jobs=joblib.cpu_count()//2):
        super().__init__()
        self.l2star = l2        # actually l2* to accommodate for frequency scaling factor v
        self.v = v              # frequency scaling
        self.k = k
        self.alpha = alpha              # weight = 1 + alpha * X
        self.max_iterations = max_iterations
        self.default_context_1 = default_context_1      # whether to use ones as default context factor or zeros.
        self.n_jobs = n_jobs
        self.scale = 0.1       # initialization scale as deviation on dot product between random factors

    @property
    def m(self):
        """ amount of users """
        assert hasattr(self, "Ms"), "fit needs to be called before this property is available"
        return self.Ms[0].shape[0]

    @property
    def n(self):
        """ amount of items """
        assert hasattr(self, "Ms"), "fit needs to be called before this property is available"
        return self.Ms[1].shape[0]

    def loss(self, data: CARSData):
        if multiply(data.shape) > 1000:
            warnings.warn("Warning, loss too expensive to compute, returning 0")
            return 0

        L = 0
        for indices in itertools.product(*[range(d) for d in data.shape]):
            # print(indices)
            pred = np.sum(multiply(M[index] for M, index in zip(self.Ms, indices)))
            if data.at(indices):
                L += (1 + self.alpha) * (1 - pred) ** 2
            else:
                L += pred ** 2

        # regularization
        Lr = sum(np.inner(l2d, np.linalg.norm(M, axis=1) ** 2) for l2d, M in zip(self.l2s[:2], self.Ms[:2]))
        if self.default_context_1:
            Lr += sum(np.inner(l2d, np.linalg.norm(M - np.ones(M.shape), axis=1) ** 2) for l2d, M in zip(self.l2s[2:], self.Ms[2:]))
        else:
            Lr += sum(np.inner(l2d, np.linalg.norm(M, axis=1) ** 2) for l2d, M in zip(self.l2s[2:], self.Ms[2:]))

        # print([L, Lr])
        return L + Lr

    def initialize(self, data: CARSDataMD):
        # Initialization scale
        prod_size = 2 if self.default_context_1 else data.ndim
        scale = 1 / np.power(self.k, 1/prod_size) * self.scale

        # Factors
        self.Ms = [np.random.normal(loc=1 if self.default_context_1 and d >= 2 else 0, scale=scale, size=(s, self.k)) for d, s in enumerate(data.shape)]

        # set context factors for missing.
        default_context = np.ones(self.k) if self.default_context_1 else np.zeros(self.k)
        for M in self.Ms[2:]:
            M[0] = default_context

        frequencies = alg_util.frequencies_for_regularization_cars(data, self.alpha)
        l2 = alg_util.l2_from_scaling(frequencies, self.l2star, self.v)
        self.l2s = [l2 * np.power(frequency, self.v) for frequency in frequencies]

        return self

    def train(self, data: CARSDataMD):
        # Precompute
        MTMs = [Md.T @ Md for Md in self.Ms]

        with Parallel(n_jobs=self.n_jobs, backend='multiprocessing') as parallel:
            for it in tqdm(range(self.max_iterations)):
                for d in range(len(self.Ms)):        # dimensions
                    # constant part of computation
                    Cd = multiply(MTMs[dd] for dd in range(len(MTMs)) if dd != d)
                    Md = self._computeFactors(data, d, Cd, self.l2s[d], parallel)

                    self.Ms[d] = Md
                    MTMs[d] = Md.T @ Md

                # print("loss", self.loss(data))

        return self

    def _computeFactors(self, data, d, Cd, l2d, parallel) -> np.array:
        result = parallel(delayed(computeFactor)(data, self.Ms, Cd, d, j, self.alpha, l2d[j]) for j in range(self.Ms[d].shape[0]))

        result = np.stack(result)
        return result

    def setUserItemFactors(self, P, Q):
        self.Ms[0] = P
        self.Ms[1] = Q
        return self

    def computeContextFactors(self, data: CARSData):
        MTMs = [Md.T @ Md for Md in self.Ms]

        with Parallel(n_jobs=self.n_jobs, backend='multiprocessing') as parallel:
            for it in tqdm(range(self.max_iterations)):
                for d in range(2, len(self.Ms)):        # dimensions
                    # constant part of computation
                    Cd = multiply(MTMs[dd] for dd in range(len(MTMs)) if dd != d)
                    Md = self._computeFactors(data, d, Cd, self.l2s[d], parallel)

                    self.Ms[d] = Md
                    MTMs[d] = Md.T @ Md

                # print("loss", self.loss(data))

        return self

    def predict_all(self, val_user_ids: np.array, val_context_indices: List[np.array]) -> np.ndarray:
        # Checking
        assert hasattr(self, "Ms"), "fit needs to be called before predict"
        assert val_user_ids.shape[0] == len(val_context_indices[0]), "users in and out need to correspond"

        # Isolate Q for faster predictions
        P, Q, *Ms = self.Ms

        factors = indexFactors([P, *Ms], [val_user_ids, *val_context_indices])
        P = multiply(factors)

        scores = P @ Q.T

        return scores


class iTALSs(iTALS):
    """
    Flattened version of iTALS. The 's' stands for 'stacked'.
    Dimensions are limited to 3 and contexts are stacked.
    """
    context_shape_md: np.array

    def fit(self, data: CARSDataMD):
        self.context_shape_md = data.context_shape
        data = data.withFlattenedContexts()
        return super().fit(data)

    def initialize(self, data: CARSDataMD):
        if isinstance(data, CARSDataMD):
            self.context_shape_md = data.context_shape
            data = data.withFlattenedContexts()
        super().initialize(data)

        # special case for missing context values
        default_context = np.ones(self.k) if self.default_context_1 else np.zeros(self.k)
        for c in range(data.shape[2]):
            if data.convertFlatContextValueToMD(c)[1] == 0:
                self.Ms[2][c] = default_context

        return self

    def train(self, data: CARSDataMD):
        if isinstance(data, CARSDataMD):
            self.context_shape_md = data.context_shape
            data = data.withFlattenedContexts()
        super().train(data)
        return self

    def computeContextFactors(self, data: CARSData):
        if isinstance(data, CARSDataMD):
            self.context_shape_md = data.context_shape
            data = data.withFlattenedContexts()
        return super().computeContextFactors(data)

    def predict_all(self, val_user_ids: np.array, val_context_indices: List[np.array]) -> np.ndarray:
        n_context_dims = len(val_context_indices)
        (val_user_ids_stacked,), val_context_indices = stackForContexts([val_user_ids], val_context_indices, self.context_shape_md)
        scores_stacked = super().predict_all(val_user_ids_stacked, [val_context_indices])

        return unstackAverageContexts(scores_stacked, val_user_ids.shape[0], self.n, n_context_dims)


def _computeFactor(indices, w, Ms, Cd, bd, l2, d) -> np.array:
    Ms_pos = indexFactors(Ms, indices)

    V = multiply(Ms_pos)
    k = V.shape[1]
    VTW = V.T * w
    A = (Cd + VTW @ V) + l2 * np.identity(k)
    # for RHS we need 1 + w, which is computed with VTW + V.T
    b = l2 * bd + (VTW + V.T).sum(axis=1)

    return np.linalg.solve(A, b)


def computeFactor(data, Ms, Cd, d, j, alpha, l2):
    k = Ms[0].shape[1]
    # get the 'default' value of factors in this dimension
    bd = Ms[d][0] if d >= 2 else np.zeros(k)

    # context factors have default values for missing that shouldn't be learned
    if d >= 2:
        # first factor is for default
        if j == 0 or (isinstance(data, CARSDataFlat) and data.convertFlatContextValueToMD(j)[1] == 0):
            return bd

    rows, indices = data.unfold(d, j)
    indices[d] = None
    w = alpha * np.ones(rows.shape[0])
    return _computeFactor(indices, w, Ms, Cd, bd, l2, d)
