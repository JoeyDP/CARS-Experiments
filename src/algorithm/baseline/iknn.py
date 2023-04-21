import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity

from ..algorithm import Algorithm


class ItemKNN(Algorithm):
    def __init__(self, k: int = 200, normalize=False):
        super().__init__()
        self.k = k
        self.normalize = normalize

    def fit(self, X: scipy.sparse.csr_matrix, S: scipy.sparse.csr_matrix = None, tags = None):
        # Input checking
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"

        m, n = X.shape

        item_cosine_similarities_ = cosine_similarity(X.T, dense_output=True)

        # Set diagonal to 0, because we don't want to support self similarity
        np.fill_diagonal(item_cosine_similarities_, 0)

        if self.k:
            top_k_per_row = np.argpartition(item_cosine_similarities_, -self.k, axis=1)[:, -self.k:]
            values = np.take_along_axis(item_cosine_similarities_, top_k_per_row, axis=1)

            res = scipy.sparse.lil_matrix(item_cosine_similarities_.shape)
            np.put_along_axis(res, top_k_per_row, values, axis=1)
            item_cosine_similarities_ = res.tocsr()

        if self.normalize:
            # normalize per row
            row_sums = np.asarray(item_cosine_similarities_.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1
            diag_div_matrix = scipy.sparse.diags(1 / row_sums)
            item_cosine_similarities_ = diag_div_matrix @ item_cosine_similarities_

        self.B_ = scipy.sparse.csr_matrix(item_cosine_similarities_)

        return self

    def predict_all(self, X: scipy.sparse.csr_matrix):
        """ Compute scores for a matrix of users (for offline evaluation) """
        # Input checking
        assert hasattr(self, "B_"), "fit needs to be called before predict"
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"

        density = self.B_.getnnz() / np.prod(self.B_.shape)
        print("density of model", density)

        scores = (X @ self.B_).toarray()

        return scores


class ItemKNNSparse(ItemKNN):
    def fit(self, X: scipy.sparse.csr_matrix, S: scipy.sparse.csr_matrix = None, tags = None):
        # Input checking
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"

        m, n = X.shape

        item_cosine_similarities_ = cosine_similarity(X.T, dense_output=False).tocsr()

        # density = item_cosine_similarities_.getnnz() / (n * n)
        # print("density", density)

        # Set diagonal to 0, because we don't want to support self similarity
        item_cosine_similarities_.setdiag(0)

        # might not be faster, but it can speed up top_k per row selection
        item_cosine_similarities_.eliminate_zeros()
        sim = item_cosine_similarities_

        if self.k:
            for row in range(n):
                start, end = sim.indptr[row], sim.indptr[row + 1]
                if end - start <= self.k:
                    continue

                values = sim.data[start:end]
                not_top_k_indices = np.argpartition(values, -self.k)[:-self.k]
                sim.data[start + not_top_k_indices] = 0

        item_cosine_similarities_.eliminate_zeros()
        item_cosine_similarities_.sort_indices()

        if self.normalize:
            # normalize per row
            row_sums = np.asarray(item_cosine_similarities_.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1
            diag_div_matrix = scipy.sparse.diags(1 / row_sums)
            item_cosine_similarities_ = diag_div_matrix @ item_cosine_similarities_

        self.B_ = item_cosine_similarities_

        return self


from tqdm.auto import tqdm

class ItemKNNIterative(ItemKNN):
    """ Reduce memory requirement by taking top K per row immediately. """
    def fit(self, X: scipy.sparse.csr_matrix, S: scipy.sparse.csr_matrix = None, tags = None):
        # Input checking
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"
        X = X.astype(np.int32)

        m, n = X.shape

        norms = np.sqrt(np.asarray(X.sum(axis=0)).flatten())
        safe_norms = norms.copy()
        safe_norms[safe_norms == 0] = 1
        diag_div = scipy.sparse.diags(1 / safe_norms)
        X = X @ diag_div

        XT = X.T.tocsr()

        data = list()
        row_ind = list()
        col_ind = list()
        for i in tqdm(range(n)):
            if norms[i] == 0:
                continue

            num = (XT[i] @ X).toarray().flatten()
            num[i] = 0

            cols, = num.nonzero()
            values = num[cols]

            if self.k < len(cols):
                top_k_indices = np.argpartition(values, -self.k)[-self.k:]
                cols = cols[top_k_indices]
                values = values[top_k_indices]

            if self.normalize:
                total = values.sum()
                if total == 0:
                    total = 1   # safe divide
                values = values / total

            col_ind.append(cols)
            rows = np.repeat(i, len(cols))
            row_ind.append(rows)
            data.append(values)

        data = np.concatenate(data, axis=0)
        row_ind = np.concatenate(row_ind, axis=0)
        col_ind = np.concatenate(col_ind, axis=0)
        sim = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, n), dtype=np.float32)

        self.B_ = sim

        return self
