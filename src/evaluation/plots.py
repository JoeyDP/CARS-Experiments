import numpy as np
from matplotlib import pyplot as plt

import src.util as util


def get_rec_counts(algorithm, histories, k=100):
    """ Computes how many times each item was recommended in top k. """
    item_rec_counts = np.zeros(histories.shape[1])

    predictions = algorithm.predict_all(histories)
    predictions[histories.astype(bool).toarray()] = -10000
    recommendations = util.predictions_to_recommendations(predictions, top_k=k)
    for u in range(recommendations.shape[0]):
        item_rec_counts[recommendations[u]] += 1

    return item_rec_counts


def plot_rec_counts_l(*item_rec_counts_l, labels=[], k=100):
    """ Helper function to plot recommendation counts. See plot_long_tail. """
    n = 0
    for index, item_rec_counts in enumerate(item_rec_counts_l):
        n = max(n, item_rec_counts.shape[0])
        values = np.sort(item_rec_counts)[::-1]
        values[values == 0] = np.nan
        values = np.log(values)
        label = ""
        if index < len(labels):
            label = labels[index]
        plt.scatter(np.arange(values.shape[0]), values, s=1, label=label)

    plt.xlim(0, n)
    plt.xlabel('Item rank')
    plt.ylabel(f"log of recommendation count in top {k}")
    plt.legend()
    plt.show()


def plot_long_tail(*algs, histories, Xtest_out=None, labels=[], k=20):
    """ Plots the 'long tail' distribution of recommendation for multiple algorithms. """
    item_rec_counts_l = [get_rec_counts(alg, histories, k=k) for alg in algs]
    if Xtest_out is not None:
        test_counts = np.asarray(Xtest_out.sum(axis=0)).flatten()
        item_rec_counts_l.append(test_counts)

    plot_rec_counts_l(*item_rec_counts_l, labels=labels, k=k)
