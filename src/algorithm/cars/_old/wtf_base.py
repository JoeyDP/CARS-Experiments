from typing import List

import numpy as np
import scipy.sparse

from tqdm.auto import tqdm

from .cars_algorithm import CARSAlgorithm
from src.data.cars import CARSData


class WTFBase(CARSAlgorithm):
    def __init__(self, l2: float = 200, k=20, alpha: float = 10, max_iterations=10):
        super().__init__()
        self.l2 = l2
        self.k = k
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.dtype = np.float64

    def fit(self, data: CARSData):
        """
        Learn the model for context aware recommendation.
        The final stored Bs differ from the ones in the derivation in that the identity matrix is subtracted from them.
        This facilitates faster predictions since the scores of contexts are added to the base prediction.

        :param Xs: List of interaction matrices per context. First matrix denotes interaction where no context is given.
        :return: self
        """

        # TODO: change to indexed learning rather than Xs notation
        Xs = [data.withoutContext().toCSR()]
        for c in data.contexts:
            Xs.append(data.withContext(c).toCSR())

        # Input checking
        for Xc in Xs:
            Xc.eliminate_zeros()
            assert np.all(Xc.data == 1), "X should only contain binary values"

        # X = self.XfromXs(Xs)
        Xs = [Xc.tocsc() for Xc in Xs]

        m, n = Xs[0].shape
        amtC = len(Xs)

        # First interaction matrix indicates no context -> we do not learn B for it and leave it at identity matrix.
        B0 = np.identity(self.k)
        Bs = [B0] + [np.identity(self.k) for c in range(amtC)]
        P = np.random.standard_normal((m, self.k)).astype(self.dtype)
        Pold = P

        # for it in tqdm(range(self.max_iterations)):
        for it in tqdm(range(self.max_iterations)):
            Q = self.computeItemFactors(Xs, P, Bs)
            P = self.computeItemFactors([Xc.T.tocsc() for Xc in Xs], Q.T, [Bc.T for Bc in Bs]).T
            # First interaction matrix indicates no context -> we do not learn B for it and leave it at identity matrix.
            Bs = [B0] + self.computeContextTransforms(Xs[1:], P, Q, Bs[1:])

            print("diff P", np.linalg.norm(P - Pold), ", rel diff P", np.linalg.norm(P-Pold)/np.linalg.norm(Pold) )
            print("norm Q", np.linalg.norm(Q))
            print("Bs[1]", np.linalg.norm(Bs[1]))
            print("-----------------------------")
            Pold = P

        self.P = P
        # Loss regularizes Bc - I, which is what we need for predictions when B0 is added (offset modelling)
        # B0 (identity) is not stored explicitly in Bs
        self.Bs = [Bc - B0 for Bc in Bs[1:]]
        self.Q = Q

        return self

    def predict_all(self, val_user_ids: np.array, val_context_indices: List[np.array]) -> scipy.sparse.csr_matrix:
        # Checking
        assert hasattr(self, "P"), "fit needs to be called before predict"
        assert val_user_ids.shape[0] == val_context_indices[0].shape[0], "users in and out need to correspond"
        
        # # First compute user factors based on interactions.
        # # If weak generalization was used this is technically not needed, but we keep it like this for simplicity.
        # # offset matrices are stored (with I subtracted) -> add identity back for learning user factors
        # B0 = np.identity(self.k)
        # Bs = [B0] + [(Bc + B0).T for Bc in self.Bs]
        # P = self.computeItemFactors([Xc.T.tocsc() for Xc in Xs], self.Q.T, Bs).T
        # del Bs

        P = self.P[val_user_ids, :].copy()

        # Then scores are computed per user because of their unique contexts.
        # We modify user factors in place based on context to more quickly compute the predictions of all items in parallel.
        B0 = np.identity(self.k)
        for row in range(P.shape[0]):
            contexts = val_context_indices[row]
            Bc = B0 + sum([self.Bs[c] for c in contexts])
            P[row] = P[row] @ Bc

        scores = P @ self.Q

        # TODO: include switch to compute scores with geometric mean instead

        return scores

    def computeItemFactor(self, Xis, PBs, BsTPTPBsl2):
        A = BsTPTPBsl2.copy() / self.alpha
        b = np.zeros(self.k, dtype=self.dtype)

        for Xic, PBc in zip(Xis, PBs):
            PBc_Xic = PBc[Xic.astype(np.bool)]
            A += PBc_Xic.T @ PBc_Xic
            b += PBc_Xic.T.sum(axis=1)

        A *= self.alpha
        b *= self.alpha
        return np.linalg.solve(A, b)

    def computeItemFactors(self, Xs, P, Bs):
        m, n = Xs[0].shape

        PBs = [P @ Bc for Bc in Bs]
        BsTPTPBsl2 = sum(PBc.T @ PBc for PBc in PBs) + self.l2 * np.identity(self.k, dtype=self.dtype)

        Q = np.zeros((self.k, n), dtype=self.dtype)
        # for i in tqdm(range(n), leave=False):
        for i in range(n):
            Xis = [Xc[:, i].toarray().flatten() for Xc in Xs]
            Q[:, i] = self.computeItemFactor(Xis, PBs, BsTPTPBsl2)

        return Q

    def computeContextTransform(self, Xc, P, Q, QQTkronPTP, Bc_old):
        m, n = Xc.shape

        A = QQTkronPTP.copy() / self.alpha
        for i in range(n):
            Qi = Q[:, i]
            Xci = Xc[:, i].toarray().flatten()

            P_Xci = P[Xci.astype(np.bool)]

            A += np.kron(np.outer(Qi, Qi), P_Xci.T @ P_Xci)

        A *= self.alpha
        b = vec(self.alpha * P.T @ Xc @ Q.T + self.l2 * np.identity(self.k))

        return unvec(np.linalg.solve(A, b))

    def computeContextTransforms(self, Xs, P, Q, Bs_old):
        """ Compute with kronecker product and inverse. """
        QQT = Q @ Q.T
        PTP = P.T @ P
        QQTkronPTP = np.kron(QQT, PTP) + self.l2 * np.identity(self.k * self.k)

        Bs = list()
        for Xc, Bc_old in tqdm(zip(Xs, Bs_old), leave=False, total=len(Xs), desc='context'):
        # for Xc, Bc_old in zip(Xs, Bs_old):
            Bc = self.computeContextTransform(Xc, P, Q, QQTkronPTP, Bc_old)
            Bs.append(Bc)

        return Bs

