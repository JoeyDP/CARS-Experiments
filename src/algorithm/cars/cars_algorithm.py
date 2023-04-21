from typing import List

import numpy as np
import scipy.sparse

from ..algorithm import Algorithm
from src.data.cars import CARSData


class CARSAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()

    def fit(self, data: CARSData):
        """
        Learn the model.

        :param data: object of CARSData
        :return: self
        """
        self.initialize(data)
        self.train(data)
        return self

    def initialize(self, data: CARSData):
        return self

    def train(self, data: CARSData):
        return self

    def setUserItemFactors(self, P, Q):
        raise NotImplementedError()

    def computeContextFactors(self, data: CARSData):
        raise NotImplementedError()

    def predict_all(self, val_user_ids: np.array, val_context_indices: List[np.array]) -> np.ndarray:
        """
        Generate recommendation scores for a batch of users given one hold out item and its context(s).

        :param val_user_ids: array of indices for the users in the training data that are used for testing (weak generalization)
        :param val_context_indices: indices of context dimension(s).
        :return sparse matrix of scores. Rows correspond to follow val_user_ids:
        """
        raise NotImplementedError()
