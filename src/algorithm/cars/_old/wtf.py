import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from tqdm.auto import tqdm

from .wtf_base import WTFBase, vec, unvec
from .wtf_numba import WTFNumba


class WTF(WTFNumba):
    """ Optimized version with conjugate gradient """
    def __init__(self, l2: float = 200, k=20, alpha: float = 10, max_iterations=10,
                 cg_steps=3):
        super().__init__(l2=l2, k=k, alpha=alpha, max_iterations=max_iterations)
        self.cg_steps = cg_steps

    def computeContextTransform(self, Xc, P, Q, QQTkronPTP, Bc_old):
        m, n = Xc.shape

        b = vec(self.alpha * P.T @ Xc @ Q.T + self.l2 * np.identity(self.k))

        PTP = P.T @ P
        QQT = Q @ Q.T

        def matvec(Bc_vec):
            # print("matvec")
            Bc = unvec(Bc_vec)
            V = (PTP @ Bc @ QQT + self.l2 * Bc) / self.alpha

            # precompute P @ Bc outside loop
            PBc = P @ Bc

            # TODO: might use JAX vmap?
            #  or might use standard parallel loop that computer LHS of outer product and then do matrix mult
            vectors = list()
            # for i in tqdm(range(n), leave=False, desc='matvec'):
            for i in range(n):
                Qi = Q[:, i]
                Xci = Xc[:, i].toarray().flatten()

                Xci_mask = Xci.astype(np.bool)
                P_Xci = P[Xci_mask]
                Pbc_Xci = PBc[Xci_mask]

                # we store the list of vectors and the compute all outer products at once with matrix multiplication
                vectors.append(P_Xci.T @ (Pbc_Xci @ Qi))

            # sum of outer products is same as matrix multiplication
            F = np.stack(vectors).T
            V += F @ Q.T

            V *= self.alpha
            res = vec(V)
            return res

        A = scipy.sparse.linalg.LinearOperator((self.k*self.k, self.k*self.k), dtype=self.dtype, matvec=matvec)

        Bc0 = vec(Bc_old).astype(self.dtype)
        Bc_vec, info = scipy.sparse.linalg.cg(A, b.astype(self.dtype), x0=Bc0, maxiter=self.cg_steps)

        return unvec(Bc_vec)

    def computeContextTransforms(self, Xs, P, Q, Bs_old):
        """ Compute with linear operator and CG """
        QQTkronPTP = None   # not needed if using CG

        Bs = list()
        for Xc, Bc_old in tqdm(zip(Xs, Bs_old), leave=False, total=len(Xs), desc='context'):
        # for Xc, Bc_old in zip(Xs, Bs_old):
            Bc = self.computeContextTransform(Xc, P, Q, QQTkronPTP, Bc_old)
            Bs.append(Bc)

        return Bs

