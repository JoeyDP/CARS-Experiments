from pathlib import Path

import pandas as pd
import numpy as np
import scipy.sparse

csr_matrix = scipy.sparse.csr_matrix

from src.data.cars import CARSDataMD


#################################
#   Save/load/parse utilities   #
#################################


def parse_interactions(interactions: Path, item_id, user_id, shape_items=0):
    """ Load interactions from csv to csr_matrix. """
    data = pd.read_csv(interactions)

    X = interactions_df_to_csr(data, item_id, user_id, shape_items=shape_items)
    return X


def parse_metadata(metadata: Path, item_id):
    """ Load metadata from csv to csr_matrix. Returns values and labels """
    data = pd.read_csv(metadata).set_index(item_id).sort_index()

    values = data.values.astype(np.int8)
    S = scipy.sparse.csr_matrix(values, dtype=np.int8)
    return S, list(data.columns)


def store_interactions(X: csr_matrix, path: Path, item_id: str, user_id: str):
    """ Write interactions to csv file. """
    rows, cols = X.nonzero()
    df = pd.DataFrame(data={user_id: rows, item_id: cols})
    df.to_csv(path, index=False)


def interactions_df_to_csr(interactions: pd.DataFrame, item_id, user_id, shape_items=0, shape_users=0) -> csr_matrix:
    """ Converts a pandas dataframe to user-item csr matrix. """
    if len(interactions) == 0:
        return scipy.sparse.csr_matrix((shape_users, shape_items))

    max_user = max(interactions[user_id].max() + 1, shape_users)
    max_item = max(interactions[item_id].max() + 1, shape_items)

    values = np.ones(len(interactions), dtype=np.int32)
    X = scipy.sparse.csr_matrix((values, (interactions[user_id], interactions[item_id])), dtype=np.int32, shape=(max_user, max_item))

    # duplicates are added together -> make binary again
    X.sum_duplicates()
    X[X > 1] = 1

    return X


def parse_interactions_with_context(interactions: Path, item_id, user_id) -> CARSDataMD:
    """ Load interactions from csv to dataframe and put in CARSData object.
        Dataframe has columns of user and item id renamed to 'userId' and 'itemId'.
        Other columns are context indices where 0 means missing.
    """
    data = pd.read_csv(interactions).rename(columns={item_id: 'itemId', user_id: 'userId'})
    return CARSDataMD(data)
