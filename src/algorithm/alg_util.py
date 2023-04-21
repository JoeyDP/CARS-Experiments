import numpy as np

from src.data.cars import CARSData
from src.util import multiply


def l2_from_scaling(frequencies, l2star, v):
    vstar = 0       # follow principle of ALS l2 scaling (l2 typically > 1)
    return l2star * np.sum(np.power(frequency, vstar).sum() for frequency in frequencies) / np.sum(np.power(frequency, v).sum() for frequency in frequencies)


def frequencies_for_regularization(X, alpha):
    return list(np.asarray(X.shape[(d+1) % 2] + X.sum(axis=(d+1)%2) * alpha).flatten() for d in range(2))


def frequencies_for_regularization_cars(data: CARSData, alpha):
    unknowns_per_dim = list(multiply(s for dd, s in enumerate(data.shape) if dd != d) for d in range(data.ndim))
    return list(np.asarray(unknowns + frequencies * alpha).flatten() for unknowns, frequencies in zip(unknowns_per_dim, data.dimension_frequencies()))

