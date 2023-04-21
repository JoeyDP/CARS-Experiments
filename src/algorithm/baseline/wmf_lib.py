import numpy as np
import scipy.sparse

from ..algorithm import Algorithm

from implicit.als import AlternatingLeastSquares


class WMF(Algorithm):
    def __init__(self, l2: float = 200, k=200, alpha=10, max_iterations=10):
        super().__init__()
        self.l2 = l2
        self.k = k
        self.alpha = alpha
        self.max_iterations = max_iterations

    def fit(self, X: scipy.sparse.csr_matrix, S: scipy.sparse.csr_matrix = None, tags = None):
        # Input checking
        X.eliminate_zeros()
        X = X.astype(np.int32)
        assert np.all(X.data == 1), "X should only contain binary values"

        # initialize a model
        model = AlternatingLeastSquares(factors=self.k, alpha=self.alpha, regularization=self.l2, iterations=self.max_iterations)

        model.fit(X)

        self.model = model

        return self

    def predict_all(self, X: scipy.sparse.csr_matrix):
        """ Compute scores for a matrix of users (for offline evaluation) """
        # Input checking
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"

        m, n = X.shape

        userids = np.arange(m)
        ranked_list, ranked_scores = self.model.recommend(userids, X, N=n, recalculate_user=True, filter_already_liked_items=False)
        scores = np.zeros(X.shape)
        np.put_along_axis(scores, ranked_list, ranked_scores, axis=1)
        return scores
