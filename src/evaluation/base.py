from functools import partial
from collections.abc import Iterable
from collections import defaultdict
import random

import numpy as np
from tqdm.auto import tqdm

from src.algorithm.algorithm import Algorithm
from .metrics import recall_k, rr_k, ndcg_k
from . import splits


def iterate_hyperparams(hyperparameter_ranges):
    combinations = [dict()]
    for param, values in hyperparameter_ranges.items():
        if not isinstance(values, Iterable):
            values = [values]
        new_combinations = list()
        for value in values:
            new_combinations.extend([{**combination, param: value} for combination in combinations])
        combinations = new_combinations
    return combinations


def eval(alg: Algorithm, Xval_in, Xval_out, retarget=False):
    predictions = alg.predict_all(Xval_in)

    if not retarget:
        # Prevent interactions from the history from being recommended again
        indices = Xval_in.nonzero()
        predictions[indices] = -1e10

    metrics = {
        'MRR@5': partial(rr_k, top_k=5),
        'MRR@20': partial(rr_k, top_k=20),
        'Average Recall@5': partial(recall_k, top_k=5),
        'Average Recall@20': partial(recall_k, top_k=20),
    }

    # metrics = {
    #     'Recall@20': partial(recall_k, top_k=20),
    #     'Recall@100': partial(recall_k, top_k=100),
    #     'NDCG@100': partial(ndcg_k, top_k=100),
    # }

    print(f"Evaluating with {Xval_out.shape[0]} users")
    scores = dict()
    for name, metric in metrics.items():
        values = metric(predictions, Xval_out)
        value = np.average(values)
        scores[name] = value

    return scores


def printMetrics(scores):
    for name, score in scores.items():
        print(name, np.around(score, decimals=3))


def gridsearch(Alg, Xtrain, Xval_in, Xval_out, hyperparameter_ranges, fit_params=dict(), retarget=False):
    best = (0, None)
    for hyperparameters in tqdm(iterate_hyperparams(hyperparameter_ranges)):
        tqdm.write(f"Training model {Alg.__name__} with hyperparameters {hyperparameters}")
        alg = Alg(**hyperparameters)
        alg.fit(Xtrain, **fit_params)
        scores = eval(alg, Xval_in, Xval_out, retarget=retarget)
        printMetrics(scores)
        target_metric_score = scores['MRR@20']
        # target_metric_score = scores['NDCG@100']
        payload = (target_metric_score, hyperparameters)
        best = max(best, payload, key=lambda x: x[0])
    return best


def printKfoldsMetrics(all_scores):
    for name, scores in all_scores.items():
        average = np.average(scores)
        std = np.std(scores, ddof=1)
        print(name, f"{np.around(average, decimals=3)} ({np.around(std, decimals=3)})")


def kFoldsEval(alg, data, nr_folds=5, seed=None, retarget=False):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    samplingSeeds = [np.random.randint(np.iinfo(np.int32).max) for _ in range(nr_folds)]

    all_scores = defaultdict(list)
    for samplingSeed in samplingSeeds:
        Xtrain, Xval_in, Xval_out = splits.leave_one_out_split_non_context(data, seed=samplingSeed)
        alg.fit(Xtrain)
        scores = eval(alg, Xval_in, Xval_out, retarget=retarget)
        for name, score in scores.items():
            all_scores[name].append(score)

    printKfoldsMetrics(all_scores)

    return all_scores
