import random
from typing import Tuple, List

import numpy as np
import scipy.sparse

from src.data.cars import CARSDataMD

csr_matrix = scipy.sparse.csr_matrix


def strong_generalization_split(X: csr_matrix, test_users: int, perc_history: float, min_interactions: int = 4,
                                seed: int = 42) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """ Splits interaction matrix X in three parts: training, val_in and val_out with strong generalization.
    Users in training and validation are disjoint.
    """
    # set seed for reproducability
    np.random.seed(seed)
    users = X.shape[0]
    assert users > test_users, "There should be at least one train user left"

    # pick users with at a certain amount of interactions
    active_users = np.where(X.sum(axis=1) >= min_interactions)[0]
    if len(active_users) < test_users:
        raise ValueError(f"Can't select {test_users} test users. There are only {len(active_users)} users with at least {min_interactions} interactions.")

    test_user_ids = np.random.choice(active_users, test_users, replace=False)
    test_user_mask = np.zeros(users, dtype=bool)
    test_user_mask[test_user_ids] = 1
    train, val = X[~test_user_mask], X[test_user_mask]
    train.eliminate_zeros()

    val_in = val.copy()
    for u in range(val_in.shape[0]):
        items = val[u].nonzero()[1]
        amt_out = int(len(items) * (1 - perc_history))
        amt_out = max(1, amt_out)                   # at least one test item required
        amt_out = min(len(items) - 1, amt_out)      # at least one train item required
        items_out = np.random.choice(items, amt_out, replace=False)

        val_in[u, items_out] = 0

    val_in.eliminate_zeros()

    val_out = val
    val_out[val_in.astype(bool)] = 0
    val_out.eliminate_zeros()

    return train, val_in, val_out


def context_leave_one_out_split(data: CARSDataMD, seed: int = None) -> Tuple[CARSDataMD, CARSDataMD]:
    """
    Perform weak generalization leave-one-out split for context aware recommendation.
    Assumes all users have at least two interactions to perform a valid train/test split.

    :param data: has dataframe with index with names 'userId' and 'itemId'. Columns are binary indicators for contexts.
    :param seed: RNG seed for reproducibility (optional)
    :return: Two CARSData objects: first one contains the training data, the second one the held out test interactions
     and their contexts. Histories of test data need to be fetched from training data (weak generalization).
    """
    # set seed for reproducability if given. Otherwise assume seed already set by caller.
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    samplingSeed = np.random.randint(np.iinfo(np.int32).max)

    df = data.df

    user_items = df.groupby('userId')  # .agg(tuple).applymap(list)
    test_interactions = user_items.sample(1, random_state=samplingSeed)
    train_interactions = df[~df.index.isin(test_interactions.index)]

    # Code below drop all interactions with left out item rather than only the interaction itself.
    # train_interactions = df[
    #     ~df.set_index(['userId', 'itemId']).index.isin(test_interactions.set_index(['userId', 'itemId']).index)]

    train_data = data._subset(train_interactions)
    test_data = data._subset(test_interactions)
    return train_data, test_data


def leave_one_out_split_non_context(data: CARSDataMD, seed: int = None) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """ Perform same splitting as context_leave_one_out_split but drop context data and convert to matrices. """
    train, test = context_leave_one_out_split(data, seed)

    Xtrain = train.toCSR()
    rows, (val_user_ids, val_item_ids, *val_context_indices) = test.unfold()
    Xval_in = Xtrain[val_user_ids]
    Xval_out = test.toCSR()[val_user_ids]
    return Xtrain, Xval_in, Xval_out
