import warnings
from pathlib import Path

import numpy as np
import scipy.sparse

from tqdm.auto import tqdm
import joblib
from joblib import Parallel, delayed

from src.algorithm.algorithm import Algorithm
from src.algorithm import alg_util


class WMF(Algorithm):
    P: np.array
    Q: np.array
    _QTQ: np.array
    l2: float
    l2u: np.array
    l2i: np.array

    def __init__(self, l2: float = 1, v=0,  k=20, alpha: float=1, max_iterations=10, n_jobs=joblib.cpu_count()//2):
        super().__init__()
        self.l2star = l2       # actually l2* to accommodate for frequency scaling factor v
        self.v = v              # frequency scaling
        self.k = k
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.n_jobs = n_jobs
        self.scale = 0.1        # initialization scale as deviation on dot product between random factors
        self._QTQ = None

    @property
    def QTQ(self):
        if self._QTQ is None:
            self._QTQ = self.Q.T @ self.Q
        return self._QTQ

    def loss(self, X: scipy.sparse.csr_matrix):
        if (X.shape[0] * X.shape[1]) > 1000:
            warnings.warn("Warning, loss too expensive to compute, returning 0")
            return 0

        Wsq = np.sqrt(np.ones(X.shape) + self.alpha * X)

        Ls = np.linalg.norm(np.multiply(Wsq, (X - self.P @ self.Q.T))) ** 2

        Lr = np.inner(self.l2u, np.linalg.norm(self.P, axis=1) ** 2)\
             + np.inner(self.l2i, np.linalg.norm(self.Q, axis=1) ** 2)

        return Ls + Lr

    def initialize(self, X: scipy.sparse.csr_matrix):
        m, n = X.shape
        scale = 1 / np.power(self.k, 1/2) * self.scale

        # Factors
        self.P = np.random.normal(loc=0, scale=scale, size=(m, self.k))
        self.Q = np.random.normal(loc=0, scale=scale, size=(n, self.k))
        self._QTQ = None

        frequencies = alg_util.frequencies_for_regularization(X, self.alpha)
        self.l2 = alg_util.l2_from_scaling(frequencies, self.l2star, self.v)
        self.l2u = self.l2 * np.power(frequencies[0], self.v)
        self.l2i = self.l2 * np.power(frequencies[1], self.v)

        return self

    def train(self, X: scipy.sparse.csr_matrix):
        # Input checking
        X.eliminate_zeros()
        X = X.astype(np.int32)
        assert np.all(X.data == 1), "X should only contain binary values"

        # invalidate cached version
        self._QTQ = None

        # Precompute
        QTQ = self.Q.T @ self.Q

        with Parallel(n_jobs=self.n_jobs, backend='multiprocessing') as parallel:
            for it in tqdm(range(self.max_iterations)):
                self.P = self._computeFactors(X, self.Q, QTQ, self.l2u, parallel)

                PTP = self.P.T @ self.P

                self.Q = self._computeFactors(X.T, self.P, PTP, self.l2i, parallel)
                QTQ = self.Q.T @ self.Q

                # print("loss", self.loss(X))

        return self

    def _computeFactors(self, X, M, MTM, l2s, parallel) -> np.array:
        result = parallel(delayed(computeFactor)(X, M, MTM, j, self.alpha, l2s[j]) for j in range(X.shape[0]))

        result = np.stack(result)
        return result

    def getUserItemFactors(self):
        return self.P, self.Q

    def saveUserItemFactors(self, path: Path):
        data = {
            'P': self.P,
            'Q': self.Q,
        }
        np.savez(path, **data)

    @staticmethod
    def loadUserItemFactors(path: Path):
        path = Path(path)
        # numpy auto adds this extension
        if path.suffix != "npz":
            path = path.with_suffix('.npz')

        with np.load(path, allow_pickle=True) as data:
            P = data['P']
            Q = data['Q']

            return P, Q

    def predict_all(self, X: scipy.sparse.csr_matrix):
        """ Compute scores for a matrix of users (for offline evaluation) """
        # Input checking
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"

        frequencies = alg_util.frequencies_for_regularization(X, self.alpha)
        l2u = self.l2 * np.power(frequencies[0], self.v)

        with Parallel(n_jobs=self.n_jobs, backend='multiprocessing') as parallel:
            P = self._computeFactors(X, self.Q, self.QTQ, l2u, parallel)

        return P @ self.Q.T


def _computeFactor(indices, w, M, MTM, l2) -> np.array:
    k = MTM.shape[0]
    M_pos = M[indices]

    M_pos_w = M_pos.T * w
    A = (MTM + M_pos_w @ M_pos) + l2 * np.identity(k)
    # for RHS we need 1 + w, which is computed with sum
    b = (M_pos_w + M_pos.T).sum(axis=1)

    return np.linalg.solve(A, b)


def computeFactor(X, M, MTM, j, alpha, l2):
    indices = X[j].nonzero()[1]
    w = alpha * np.ones(len(indices))
    return _computeFactor(indices, w, M, MTM, l2)
