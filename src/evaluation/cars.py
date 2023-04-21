import random
from functools import partial
from collections import defaultdict

import numpy as np
import scipy.sparse
from tqdm.auto import tqdm

from src.algorithm.cars.cars_algorithm import CARSAlgorithm
from src.data.cars import CARSData

from .metrics import recall_k, rr_k
from .base import iterate_hyperparams, printKfoldsMetrics, printMetrics
from . import splits
import src.util as util


def eval(alg: CARSAlgorithm, train: CARSData, test: CARSData, retarget=False):
    rows, (val_user_ids, val_item_ids, *val_context_indices) = test.unfold()
    # correct values in sparse format (only to be used with leave-one-out)
    Xval_out = test.toCSR()[val_user_ids]

    # gives one top-k list per left out item so can only be used with leave-one-out
    predictions = alg.predict_all(val_user_ids, val_context_indices)
    # Prevent interactions from the history from being recommended again
    if not retarget:
        X = train.toCSR()[val_user_ids]
        indices = X.nonzero()
        predictions[indices] = -1e10

    metrics = {
        'MRR@5': partial(rr_k, top_k=5),
        'MRR@20': partial(rr_k, top_k=20),
        'Average Recall@5': partial(recall_k, top_k=5),
        'Average Recall@20': partial(recall_k, top_k=20),
    }

    print(f"Evaluating with {Xval_out.shape[0]} users")
    scores = dict()
    for name, metric in metrics.items():
        values = metric(predictions, Xval_out)
        value = np.average(values)
        scores[name] = value

    return scores


def gridsearch(Alg, train: CARSData, test: CARSData, hyperparameter_ranges, fit_params=dict(), retarget=False):
    best = (0, None)
    for hyperparameters in tqdm(iterate_hyperparams(hyperparameter_ranges)):
        tqdm.write(f"Training model {Alg.__name__} with hyperparameters {hyperparameters}")
        alg = Alg(**hyperparameters)
        alg.fit(train, **fit_params)
        scores = eval(alg, train, test, retarget=retarget)
        printMetrics(scores)
        target_metric_score = scores['MRR@20']
        payload = (target_metric_score, hyperparameters)
        best = max(best, payload, key=lambda x: x[0])
    return best


def gridsearchSetFactors(Alg, train: CARSData, test: CARSData, P, Q, hyperparameter_ranges, retarget=False):
    """
    Only learns the context factors with the given algorithm. User and item factors are learned by a different model.
    """
    best = (0, None)
    for hyperparameters in tqdm(iterate_hyperparams(hyperparameter_ranges)):
        alg = Alg(**hyperparameters)
        tqdm.write(f"Training model {alg} with hyperparameters {hyperparameters}")

        alg.initialize(train)
        alg.setUserItemFactors(P, Q)
        alg.computeContextFactors(train)

        ## learn one additional iteration
        # alg.max_iterations = 1
        # alg.train(train)

        scores = eval(alg, train, test, retarget=retarget)
        printMetrics(scores)
        target_metric_score = scores['MRR@20']
        payload = (target_metric_score, hyperparameters)
        best = max(best, payload, key=lambda x: x[0])
    return best


def contextKFoldsEval(alg, data, nr_folds=5, seed=None, retarget=False):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    samplingSeeds = [np.random.randint(np.iinfo(np.int32).max) for _ in range(nr_folds)]

    all_scores = defaultdict(list)
    for samplingSeed in samplingSeeds:
        train, test = splits.context_leave_one_out_split(data, seed=samplingSeed)
        alg.fit(train)
        scores = eval(alg, train, test, retarget=retarget)
        for name, score in scores.items():
            all_scores[name].append(score)

    printKfoldsMetrics(all_scores)

    return all_scores


def contextKFoldsSetFactorsEval(alg: CARSAlgorithm, data: CARSData, userItemFactorSupplier, nr_folds=5, seed=None, retarget=False):
    """
    Only learns the context factors with the given algorithm. User and item factors are learned by a different model.
    userItemFactorSupplier is a function that takes the training data and used seed and returns the user and item factors to be used.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    samplingSeeds = [np.random.randint(np.iinfo(np.int32).max) for _ in range(nr_folds)]

    all_scores = defaultdict(list)
    for samplingSeed in samplingSeeds:
        train, test = splits.context_leave_one_out_split(data, seed=samplingSeed)

        P, Q = userItemFactorSupplier(train, samplingSeed)
        alg.initialize(train)
        alg.setUserItemFactors(P, Q)
        alg.computeContextFactors(train)

        ## learn one additional iteration
        # alg.max_iterations = 1
        # alg.train(train)

        scores = eval(alg, train, test, retarget=retarget)
        for name, score in scores.items():
            all_scores[name].append(score)

    return all_scores