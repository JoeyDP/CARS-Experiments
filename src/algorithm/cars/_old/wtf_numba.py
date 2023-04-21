import numpy as np
import scipy.linalg

from numba import jit, njit, prange
from numba.typed import List

from .wtf_base import WTFBase


class WTFNumba(WTFBase):
    def __init__(self, l2: float = 200, k=20, alpha: float = 10, max_iterations=10):
        super().__init__(l2=l2, k=k, alpha=alpha, max_iterations=max_iterations)

    def computeItemFactors(self, Xs, P, Bs):
        n = Xs[0].shape[1]
        Xs_data = List([(Xc.indptr, Xc.indices) for Xc in (Xc.tocsc() for Xc in Xs)])
        # Make sure all same order (transpose changes C to F)
        Bs = List([np.asarray(Bc, order='C') for Bc in Bs])
        factors = computeItemFactors(Xs_data, P, Bs, self.l2, self.alpha, n)

        # true_factors = super().computeItemFactors(Xs, P, Bs)
        # assert np.allclose(true_factors, factors)

        return factors


# @njit()
@njit()
def computeItemFactors(Xs_data, P, Bs, l2, alpha, n):
    """ Xs_data contains tuples of indptr, indices of csc format per context. """
    k = Bs[0].shape[0]
    PBs = List()
    BsTPTPBsl2 = l2 * np.identity(k)
    for Bc in Bs:
        PBc = P @ Bc
        BsTPTPBsl2 += PBc.T @ PBc
        PBs.append(PBc)

    Q = np.zeros((k, n))
    print("loop")
    for i in prange(n):
        Xis_indices = List()
        for Xc_indptr, Xc_indices in Xs_data:
            start, end = Xc_indptr[i], Xc_indptr[i + 1]
            indices = Xc_indices[start:end]
            Xis_indices.append(indices)
        Q[:, i] = computeItemFactor(Xis_indices, PBs, BsTPTPBsl2, alpha)

    print("loop done")
    return Q


# @jit()
@njit(inline='always')
def computeItemFactor(Xis_indices, PBs, BsTPTPBsl2, alpha):
    k = PBs[0].shape[1]
    A = BsTPTPBsl2 / alpha
    b = np.zeros(k)

    for Xic_indices, PBc in zip(Xis_indices, PBs):
        PBc_Xic = PBc[Xic_indices]
        A += PBc_Xic.T @ PBc_Xic
        b += PBc_Xic.T.sum(axis=1)

    A *= alpha
    b *= alpha
    return np.linalg.solve(A, b)
