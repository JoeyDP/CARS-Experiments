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
from src.util import diag_dot


class iTALSx(CARSAlgorithm):
    """
    iTALSx algorithm based on paper:
    Hidasi, Balázs. "Factorization models for context-aware recommendations." Infocommun J VI (4) (2014): 27-34.
    """
    Ms: List[np.array]
    context_shape_md: np.array
    l2s: List[np.array]

    def __init__(self, l2: float = 1, v=0, k=20, alpha: float = 1, max_iterations=10,
                 n_jobs=joblib.cpu_count()//2):
        super().__init__()
        self.l2star = l2        # actually l2* to accommodate for frequency scaling factor v
        self.v = v              # frequency scaling
        self.k = k
        self.alpha = alpha              # weight = 1 + alpha * X
        self.max_iterations = max_iterations
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

    @property
    def P(self):
        return self.Ms[0]

    @property
    def Q(self):
        return self.Ms[1]

    @property
    def B(self):
        return self.Ms[2]

    def loss(self, data):
        if multiply(data.shape) > 1000:
            warnings.warn("Warning, loss too expensive to compute, returning 0")
            return 0

        L = 0
        for indices in itertools.product(*[range(d) for d in data.shape]):
            # print(indices)
            vectors = [M[index] for M, index in zip(self.Ms, indices)]
            pred = np.sum(np.inner(v1, v2) for v1, v2 in itertools.combinations(vectors, 2))
            if data.at(indices):
                L += (1 + self.alpha) * (1 - pred) ** 2
            else:
                L += pred ** 2

        # regularization
        Lr = sum(np.inner(l2d, np.linalg.norm(M, axis=1) ** 2) for l2d, M in zip(self.l2s, self.Ms))

        # print([L, Lr])
        return L + Lr

    def fit(self, data: CARSDataMD):
        self.context_shape_md = data.context_shape
        data = data.withFlattenedContexts()
        return super().fit(data)

    def initialize(self, data: CARSDataMD):
        if isinstance(data, CARSDataMD):
            self.context_shape_md = data.context_shape
            data = data.withFlattenedContexts()

        # Initialization scale
        scale = 1 / np.power(self.k, 1/2) * self.scale / 3

        # Factors
        self.Ms = [np.random.normal(loc=0, scale=scale, size=(s, self.k)) for s in data.shape]

        # set context factors for missing (they aren't learned from).
        default_context = np.zeros(self.k)

        for c in range(data.shape[2]):
            if data.convertFlatContextValueToMD(c)[1] == 0:
                self.Ms[2][c] = default_context

        frequencies = alg_util.frequencies_for_regularization_cars(data, self.alpha)
        l2 = alg_util.l2_from_scaling(frequencies, self.l2star, self.v)
        self.l2s = [l2 * np.power(frequency, self.v) for frequency in frequencies]

        return self

    def train(self, data: CARSDataMD):
        if isinstance(data, CARSDataMD):
            self.context_shape_md = data.context_shape
            data = data.withFlattenedContexts()

        # m, n, l = data.shape
        s = data.shape

        # Precompute
        MTMs = [Md.T @ Md for Md in self.Ms]
        Msums = [M.sum(axis=0) for M in self.Ms]

        with Parallel(n_jobs=self.n_jobs, backend='multiprocessing') as parallel:
            for it in tqdm(range(self.max_iterations)):
                for d in range(len(self.Ms)):
                    # other two dimensions
                    e, f = (d + 1) % 3, (d - 1) % 3

                    # constant part of computation
                    Cd = np.outer(Msums[e], Msums[f])
                    Cd += Cd.T
                    Cd += s[f] * MTMs[e] + s[e] * MTMs[f]
                    bd = MTMs[e] @ Msums[f].T + MTMs[f] @ Msums[e].T

                    # parallel compute of factors
                    Md = self._computeFactors(data, d, Cd, bd, self.l2s[d], parallel)

                    # update precomputed values
                    self.Ms[d] = Md
                    MTMs[d] = Md.T @ Md
                    Msums[d] = Md.sum(axis=0)

                # print("loss", self.loss(data))

        return self

    def _computeFactors(self, data, d, Cd, bd, l2d, parallel) -> np.array:
        result = parallel(delayed(computeFactor)(data, self.Ms, Cd, bd, d, j, self.alpha, l2d[j]) for j in range(self.Ms[d].shape[0]))

        result = np.stack(result)
        return result

    def setUserItemFactors(self, P, Q):
        self.Ms[0] = P
        self.Ms[1] = Q
        return self

    def computeContextFactors(self, data: CARSData):
        if isinstance(data, CARSDataMD):
            self.context_shape_md = data.context_shape
            data = data.withFlattenedContexts()

        s = data.shape
        # Precompute
        MTMs = [Md.T @ Md for Md in self.Ms]
        Msums = [M.sum(axis=0) for M in self.Ms]

        with Parallel(n_jobs=self.n_jobs, backend='multiprocessing') as parallel:
            d = 2
            # other two dimensions
            e, f = (d + 1) % 3, (d - 1) % 3

            # constant part of computation
            Cd = np.outer(Msums[e], Msums[f])
            Cd += Cd.T
            Cd += s[f] * MTMs[e] + s[e] * MTMs[f]
            bd = MTMs[e] @ Msums[f].T + MTMs[f] @ Msums[e].T

            # parallel compute of factors
            Md = self._computeFactors(data, d, Cd, bd, self.l2s[d], parallel)

            # update precomputed values
            self.Ms[d] = Md

        return self

    def _predict_all(self, val_user_ids: np.array, val_context_indices: List[np.array]) -> np.ndarray:
        assert len(val_context_indices) == 1, "Only three dimensions supported here. Use predict_all otherwise."

        P, Q, B = self.Ms

        Psub, Bsub = indexFactors([P, B], [val_user_ids, *val_context_indices])

        # item independent part repeated for all item scores
        scores = np.tile(np.multiply(Psub, Bsub).sum(axis=1), (self.n, 1)).T
        # add item dependent part
        scores += (Psub + Bsub) @ Q.T

        return scores

    def predict_all(self, val_user_ids: np.array, val_context_indices: List[np.array]) -> np.ndarray:
        # Checking
        assert hasattr(self, "P"), "fit needs to be called before predict"
        assert val_user_ids.shape[0] == len(val_context_indices[0]), "users in and out need to correspond"

        n_context_dims = len(val_context_indices)
        (val_user_ids_stacked,), val_context_indices = stackForContexts([val_user_ids], val_context_indices, self.context_shape_md)
        scores_stacked = self._predict_all(val_user_ids_stacked, [val_context_indices])

        return unstackAverageContexts(scores_stacked, val_user_ids.shape[0], self.n, n_context_dims)


def _computeFactor(indices, weights, Ms, Cd, bd, l2) -> np.array:
    k = Ms[0].shape[1]
    Ms_pos = indexFactors(Ms, indices)

    Ms_sum = sum(Ms_pos)

    A = Cd.copy() + l2 * np.identity(k)
    A += (Ms_sum * weights[:, np.newaxis]).T @ Ms_sum

    Ms_inner = diag_dot(Ms_pos[0], Ms_pos[1].T)
    b = -bd
    b += (weights + 1 - weights * Ms_inner) @ Ms_sum

    return np.linalg.solve(A, b)


def computeFactor(data, Ms, Cd, bd, d, j, alpha, l2):
    # Cd and bd are constant parts of computation
    k = Ms[0].shape[1]
    # first factor is for default
    if d == 2 and isinstance(data, CARSDataFlat) and data.convertFlatContextValueToMD(j)[1] == 0:
        return np.zeros(k)

    rows, indices = data.unfold(d, j)
    indices[d] = None
    w = alpha * np.ones(rows.shape[0])
    return _computeFactor(indices, w, Ms, Cd, bd, l2)
