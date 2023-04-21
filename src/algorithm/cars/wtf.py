from typing import List
import itertools
import warnings

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from tqdm.auto import tqdm
import joblib
from joblib import Parallel, delayed

from .cars_algorithm import CARSAlgorithm
from src.data.cars import CARSData, CARSDataMD, CARSDataFlat, stackForContexts
from src.util import multiply, indexFactors, unstackAverageContexts, vec, unvec, diag_dot
from src.algorithm import alg_util


class WTF(CARSAlgorithm):
    P: np.array
    Q: np.array
    Bs: np.array
    default_context: np.array
    context_shape_md: np.array
    l2u: np.array
    l2i: np.array
    l2c: np.array

    def __init__(self, l2: float = 1, v=0, k=20, alpha: float = 1, max_iterations=10,
                 default_context_1=False, max_cg_iter=3,
                 n_jobs=joblib.cpu_count()//2):
        super().__init__()
        self.l2star = l2        # actually l2* to accommodate for frequency scaling factor v
        self.v = v              # frequency scaling
        self.k = k
        self.alpha = alpha              # weight = 1 + alpha * X
        self.max_iterations = max_iterations
        self.default_context_1 = default_context_1      # whether to use ones as default context factor or zeros.
        self.max_cg_iter = max_cg_iter
        self.n_jobs = n_jobs
        self.scale = 0.1       # initialization scale as deviation on dot product between random factors

    @property
    def m(self):
        """ amount of users """
        assert hasattr(self, "P"), "fit needs to be called before this property is available"
        return self.P.shape[0]

    @property
    def n(self):
        """ amount of items """
        assert hasattr(self, "Q"), "fit needs to be called before this property is available"
        return self.Q.shape[0]

    def loss(self, data: CARSDataFlat, P, Q, Bs):
        if multiply(data.shape) > 1000:
            warnings.warn("Warning, loss too expensive to compute, returning 0")
            return 0

        L = 0
        for indices in itertools.product(*[range(d) for d in data.shape]):
            u, i, c = indices
            pred = P[u] @ Bs[c] @ Q[i].T

            if data.at(indices):
                L += (1 + self.alpha) * (1 - pred) ** 2
            else:
                L += pred ** 2

        # regularization
        Lr = np.inner(self.l2u, np.linalg.norm(P, axis=1) ** 2)
        Lr += np.inner(self.l2i, np.linalg.norm(Q, axis=1) ** 2)

        if self.default_context_1:
            Lr += sum(l2 * (np.linalg.norm(Bc - np.identity(Bc.shape[1])) ** 2) for l2, Bc in zip(self.l2c, Bs))
        else:
            Lr += sum(l2 * (np.linalg.norm(Bc) ** 2) for l2, Bc in zip(self.l2c, Bs))

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

        m, n, l = data.shape

        # Initialization scale
        prod_size = 2 if self.default_context_1 else data.ndim
        scale = 1 / np.power(self.k, 1/prod_size) * self.scale

        # Factors
        self.P = np.random.normal(loc=0, scale=scale, size=(m, self.k))
        self.Q = np.random.normal(loc=0, scale=scale, size=(n, self.k))

        # set context factors for missing.
        self.default_context = np.identity(self.k) if self.default_context_1 else np.zeros((self.k, self.k))
        self.Bs = np.array([
            self.default_context +
            (np.random.normal(loc=0, scale=scale, size=(self.k, self.k)) if data.convertFlatContextValueToMD(c)[1] != 0 else 0)
            for c in range(l)])

        frequencies = alg_util.frequencies_for_regularization_cars(data, self.alpha)
        l2 = alg_util.l2_from_scaling(frequencies, self.l2star, self.v)
        self.l2u, self.l2i, self.l2c = [l2 * np.power(frequency, self.v) for frequency in frequencies]

        return self

    def train(self, data: CARSDataMD):
        if isinstance(data, CARSDataMD):
            self.context_shape_md = data.context_shape
            data = data.withFlattenedContexts()

        # Precompute
        QTQ = self.Q.T @ self.Q

        with Parallel(n_jobs=self.n_jobs, backend='multiprocessing') as parallel:
            for it in tqdm(range(self.max_iterations)):
                # oldP = P
                self.P = self._computeFactors(data, 0, self.Q, QTQ, self.Bs, self.l2u, parallel)
                PTP = self.P.T @ self.P
                # print("diff P\t\t\t\t", np.linalg.norm(oldP - P) ** 2)
                # print("norm P", np.linalg.norm(P))

                # oldQ = Q
                self.Q = self._computeFactors(data, 1, self.P, PTP, self.Bs.transpose((0, 2, 1)), self.l2i, parallel)
                QTQ = self.Q.T @ self.Q
                # print("diff Q\t\t\t\t", np.linalg.norm(oldQ - Q) ** 2)
                # print("norm Q", np.linalg.norm(Q))

                # oldBs = Bs
                self.Bs = self._computeContextFactors(data, self.P, self.Q, PTP, QTQ, self.default_context, self.Bs, self.l2c, parallel)
                # print("diff B\t\t\t\t", np.linalg.norm(oldBs - Bs) ** 2)
                # print("norm B", np.linalg.norm(Bs))

                # print("loss", self.loss(data, self.P, self.Q, self.Bs))
                # print()

        return self

    def _computeFactors(self, data, d, M, MTM, Bs, l2d, parallel) -> np.array:
        # M is either P or Q
        # precompute for all
        C = sum(Bc @ MTM @ Bc.T for Bc in Bs)
        result = parallel(delayed(computeFactor)(data, d, M, Bs, C, j, self.alpha, l2d[j]) for j in range(data.shape[d]))

        result = np.stack(result)
        return result

    def _computeContextFactors(self, data, P, Q, PTP, QTQ, defaultB, Bs, l2d, parallel):
        result = parallel(
            delayed(computeContextFactor)(data, PTP, QTQ, P, Q, defaultB, Bs[j], j, self.alpha, l2d[j], self.max_cg_iter) for j in range(data.shape[2]))

        return np.array(result)

    def setUserItemFactors(self, P, Q):
        self.P = P
        self.Q = Q
        return self

    def computeContextFactors(self, data: CARSData):
        if isinstance(data, CARSDataMD):
            self.context_shape_md = data.context_shape
            data = data.withFlattenedContexts()

        # Precompute
        QTQ = self.Q.T @ self.Q
        PTP = self.P.T @ self.P

        with Parallel(n_jobs=self.n_jobs, backend='multiprocessing') as parallel:
            self.Bs = self._computeContextFactors(data, self.P, self.Q, PTP, QTQ, self.default_context, self.Bs, self.l2c, parallel)

        return self

    def _predict_all(self, val_user_ids: np.array, val_context_indices: List[np.array]) -> np.ndarray:
        assert len(val_context_indices) == 1, "Only three dimensions supported here. Use predict_all otherwise."

        P, Bpos = indexFactors([self.P, self.Bs], [val_user_ids, *val_context_indices])

        for u in range(P.shape[0]):
            P[u] = P[u] @ Bpos[u]

        scores = P @ self.Q.T

        return scores

    def predict_all(self, val_user_ids: np.array, val_context_indices: List[np.array]) -> np.ndarray:
        # Checking
        assert hasattr(self, "Bs"), "fit needs to be called before predict"
        assert val_user_ids.shape[0] == len(val_context_indices[0]), "users in and out need to correspond"

        n_context_dims = len(val_context_indices)
        (val_user_ids_stacked,), val_context_indices = stackForContexts([val_user_ids], val_context_indices, self.context_shape_md)
        scores_stacked = self._predict_all(val_user_ids_stacked, [val_context_indices])

        return unstackAverageContexts(scores_stacked, val_user_ids.shape[0], self.n, n_context_dims)


def _computeFactor(indices, weights, M, Bs, C, l2) -> np.array:
    k = M.shape[1]
    Mpos, Bpos = indexFactors([M, Bs], indices)

    Mmod = np.zeros(Mpos.shape)
    for r in range(weights.shape[0]):
        Mj, Bc = Mpos[r], Bpos[r]
        Mmod[r] = Bc @ Mj.T

    A = C + (Mmod.T * weights) @ Mmod + l2 * np.identity(k)
    # w corresponds to w' in paper.
    b = ((weights[np.newaxis, :] + 1) @ Mmod).flatten()

    return np.linalg.solve(A, b)


def computeFactor(data, d, M, Bs, C, j, alpha, l2):
    rows, indices = data.unfold(d, j)
    del indices[d]
    w = alpha * np.ones(rows.shape[0])
    return _computeFactor(indices, w, M, Bs, C, l2)


def _computeContextFactor(indices, weights, PTP, QTQ, P, Q, defaultB, B0, l2, maxiter):
    k = PTP.shape[0]
    Ppos, Qpos = indexFactors([P, Q], indices)

    def matvec(v):
        B = unvec(v)

        Pmod = Ppos @ B
        s = diag_dot(Pmod, Qpos.T)
        R = PTP @ B @ QTQ + (Ppos * (weights * s)[:, np.newaxis]).T @ Qpos + l2 * B

        return vec(R)

    A = scipy.sparse.linalg.LinearOperator(shape=(k*k, k*k), matvec=matvec)

    b = (Ppos.T * (weights + 1)) @ Qpos + l2 * defaultB
    b = vec(b)
    # TODO: could add preconditioning M (inverse of diagonal of A works well)
    v, info = scipy.sparse.linalg.cg(A, b, x0=vec(B0), maxiter=maxiter)

    # if info > 0:
    #     print("cg did not converge")

    # print("info", info)
    B = unvec(v)
    return B


def computeContextFactor(data: CARSDataFlat, PTP, QTQ, P, Q, defaultB, B0, j, alpha, l2, maxiter):
    d, c = data.convertFlatContextValueToMD(j)
    if c == 0:
        return defaultB

    rows, indices = data.unfold(2, j)
    del indices[2]
    w = alpha * np.ones(rows.shape[0])
    return _computeContextFactor(indices, w, PTP, QTQ, P, Q, defaultB, B0, l2, maxiter)
