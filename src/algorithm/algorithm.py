import numpy as np
import scipy.sparse


class Algorithm:

    def fit(self, X: scipy.sparse.csr_matrix):
        self.initialize(X)
        self.train(X)
        return self

    def initialize(self, X: scipy.sparse.csr_matrix):
        return self

    def train(self, X: scipy.sparse.csr_matrix):
        return self

    def predict_all(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        raise NotImplementedError()


