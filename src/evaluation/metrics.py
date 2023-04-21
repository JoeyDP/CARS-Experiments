import numpy as np
import scipy.sparse

import src.util as util


def recall_k(predictions: np.ndarray, Xval_out: scipy.sparse.csr_matrix, top_k):
    """ Implements a stratified Recall@k that calculates per user:
        #correct / min(k, #user_val_items)
    """
    assert predictions.shape[0] == Xval_out.shape[0], "Predictions and ground truth need to match in shape"
    recommendations = util.predictions_to_recommendations(predictions, top_k=top_k)
    hits = np.take_along_axis(Xval_out, recommendations, axis=1)
    total_hits = np.asarray(hits.sum(axis=1)).flatten()
    best_possible = np.asarray(Xval_out.sum(axis=1)).flatten()
    best_possible[best_possible < 1] = 1
    best_possible[best_possible > top_k] = top_k
    recall_scores = total_hits / best_possible
    return recall_scores


def rr_k(predictions: np.ndarray, Xval_out: scipy.sparse.csr_matrix, top_k):
    """ Implements reciprocal rank scores. Mean needs to be taken for MRR.
    """
    assert predictions.shape[0] == Xval_out.shape[0], "Predictions and ground truth need to match in shape"
    recommendations = util.predictions_to_recommendations(predictions, top_k=top_k)

    hits = np.take_along_axis(Xval_out, recommendations, axis=1)
    ranks = np.arange(top_k) + 1
    hit_ranks = hits.multiply(ranks)

    # invert hit ranks other than 0. hit_ranks now contains inverse hit ranks
    hit_ranks.data = 1. / hit_ranks.data
    best_hit_ranks_inv = hit_ranks.max(axis=1)
    return best_hit_ranks_inv.toarray().flatten()


def ndcg_k(predictions: np.ndarray, Xval_out: scipy.sparse.csr_matrix, top_k):
    """ Implements normalized discounted cumulative gain.
    """
    assert predictions.shape[0] == Xval_out.shape[0], "Predictions and ground truth need to match in shape"
    recommendations = util.predictions_to_recommendations(predictions, top_k=top_k)

    hits = np.take_along_axis(Xval_out, recommendations, axis=1)
    ranks = np.arange(top_k) + 1
    hit_ranks = hits.multiply(ranks)

    hit_ranks.data = 1. / np.log2(hit_ranks.data + 1)
    dcg = np.asarray(hit_ranks.sum(axis=1)).flatten()

    hist_len = Xval_out.getnnz(axis=1).astype(np.int32)
    hist_len[hist_len > top_k] = top_k
    discount_template = 1. / np.log2(np.arange(2, top_k + 2))
    idcg = np.array([(discount_template[:n]).sum() for n in hist_len])

    ndcg = dcg / idcg

    # If we divide 0 by 0 -> set to 0 instead of nan
    ndcg[dcg == 0] = 0

    return ndcg