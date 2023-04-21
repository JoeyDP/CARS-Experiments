from typing import Tuple, Iterable, List, Union
from copy import deepcopy

import pandas as pd
import numpy as np
import scipy.sparse
import random
import subprocess
csr_matrix = scipy.sparse.csr_matrix

def remap_ids(*dfs, col):
    """ Maps the ids in a certain column of each dataframe to consecutive integer identifiers.
    This operation is handled inplace.
    """
    keys = set()
    for df in dfs:
        keys |= set(df[col].unique())
    id_mapping = pd.Series(np.arange(len(keys)), index=list(keys))
    for df in dfs:
        df[col] = id_mapping[df[col]].values
    return dfs


#########################
#   Numeric utilities   #
#########################

def rescale(a, lower=-1, upper=1):
    """ Rescale values in an array linearly to a certain range """
    a = a.copy()
    l, u = a.min(), a.max()
    a -= l              # positive with minimum 0
    a /= u - l          # between 0 and 1
    a *= upper - lower  # between 0 and (upper - lower)
    a += lower          # between lower and upper
    return a


def rescale0(a, neg=-1, pos=1):
    """ Rescale values in an array such that zero remains unscaled and
    highest absolute value is assumed as upper and lower bound (stretch negative and positive sides to targets). """
    a = a.copy()
    m = np.abs(a).max()

    if neg == -pos:
        a *= pos / m
    else:
        a[a < 0] *= -neg / m
        a[a > 0] *= pos / m

    return a


def normalize(a):
    """ Make array sum up to one. """
    if np.all(a == 0):
        return a
    return a / a.sum()


def diag_dot(A, B):
    """ Returns diagonal of dot product between A and B. """
    min_outer_size = min(A.shape[0], B.shape[1])

    A = A[:min_outer_size]
    B = B[:, :min_outer_size]
    if scipy.sparse.issparse(B):
        return np.asarray(np.sum(B.T.multiply(A), axis=1)).flatten()
    elif scipy.sparse.issparse(A):
        return np.asarray(np.sum(A.multiply(B.T), axis=1)).flatten()
    else:
        return np.sum(A * B.T, axis=1).flatten()


def multiply(As: Iterable[np.array]):
    """ Elementwise multiplication of matrices in list As """
    R = None
    for A in As:
        if R is None:
            R = A
        else:
            R = np.multiply(R, A)

    return R


def indexFactors(Ms: List[np.array], indices: List[np.array]):
    """ returns factors Ms indexed by indices
    :param Ms: list of factors per dimension
    :param indices: list of indices per dimension.
    Can also contain None which means the corresponding factors will be omitted from the output
    :return: factors at indices
    """
    return [Md[indices_d] for Md, indices_d in zip(Ms, indices) if indices_d is not None]
    #
    # result = list()
    # for d, indices_d in enumerate(indices):
    #     # the used dimension
    #     if indices_d is None:
    #         continue
    #
    #     Md = Ms[d]
    #
    #     result.append(Md[indices_d])
    #
    # return result


def unstackAverageContexts(scores_stacked, m, n, d):
    """ take average over all contexts of the interaction. Used when predicting for flattened data. """
    scores = np.zeros((m, n))
    for c in range(d):
        scores += scores_stacked[c * m: (c + 1) * m]

    scores /= d
    return scores


def vec(A):
    """ Vectorization of matrix by stacking columns. """
    return A.flatten('F')


def unvec(a, shape=None):
    """ Tries to perform inverse of vec operation. Either provide shape or square matrix is assumed. """
    if shape is None:
        n = np.sqrt(a.shape[0])
        assert n - int(n) == 0
        n = int(n)
        shape = (n, n)
    return a.reshape(shape, order='F')


################################
#   Recommendation utilities   #
################################

def prediction_to_recommendations(predictions: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Takes a row of user-item scores and returns a ranked list of the top_k items along with their scores. """
    if top_k == 0:
        return np.array([]), np.array([])
    recommendations = np.argpartition(predictions, -1-np.arange(top_k))[-top_k:][::-1]
    scores = predictions[recommendations]
    return recommendations, scores


def predictions_to_recommendations(predictions: np.ndarray, top_k: int) -> np.ndarray:
    """ Takes a matrix of user-item scores and returns a ranked list of the top_k items per user. """
    recommendations = np.argpartition(predictions, -1-np.arange(top_k), axis=1)[:, -top_k:][:, ::-1]
    # scores = np.take_along_axis(predictions, recommendations, axis=1)
    return recommendations

################################
#        (f)fm utilities       #
################################

def makeTrainfm(train_df, df, nbrNeg: int, outputPath, libFMPath, sparse : bool = True):
    matrix_na = train_df.df.values.tolist()
    maxValues = np.array( np.max( df.to_numpy(), axis = 0))
    maxValues = maxValues + np.ones( len(maxValues), dtype = int)
    cumSumMaxValues = np.cumsum( maxValues)
    cumSumMaxValues = np.insert(cumSumMaxValues, 0,0)

    maxItem = maxValues[1]
    matrixNeg = []
    output = ""

    # Positive interactions and make matrix with negative interactions.
    # for the negative interactions, we take a positive interaction (user, item, context) and we change only the item.
    # we take for every positive interaction, nbrNeg negative interactions.
    # ! Remark that in contextKFoldsEval_FM.py we take standard only one 1 negative interaction (as it take quite long to make a trainfile and we need to do this K times there.
    for rowMatrix in matrix_na:
        output += " 1 "
        for i in range( nbrNeg):
            rowMatrix1 = deepcopy( rowMatrix)
            rowMatrix1[1] = random.randint(0, maxItem )
            while rowMatrix1 in matrix_na:
                rowMatrix1[1] = random.randint(0, maxItem )
            matrixNeg.append( rowMatrix1)

        for (i,elem) in enumerate(rowMatrix):        
            output += str( cumSumMaxValues[i] + elem) + ":1 "
        output += "\n"

    for rowMatrix in matrixNeg:
        output += "0 "
        for (i,elem) in enumerate(rowMatrix):        
            output += str( cumSumMaxValues[i] + elem) + ":1 "
        output += "\n"
    
    with open( outputPath + '/train.libfm', 'w') as f:
        f.write(output)
    if sparse:
        command = libFMPath + "/bin/convert --ifile " + outputPath + "/train.libfm --ofilex " 

        commandOutput = outputPath + "/trainSparse.x --ofiley " + outputPath + "/trainSparse.y"
        command = command + commandOutput

        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        p.communicate()

        command = libFMPath + "/bin/transpose --ifile " + outputPath + "/trainSparse.x --ofile " + outputPath + "/trainSparse.xt"

        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        p.communicate() 
    
    return

def makeTestfm(test_df, df, outputPath, libFMPath, sparse : bool = True):
    matrix_na = test_df.df.to_numpy()
    maxValues = np.max( df.to_numpy(), axis = 0)
    maxValues = maxValues + np.ones( len(maxValues), dtype = int)
    cumSumMaxValues = np.cumsum( maxValues)
    cumSumMaxValues = np.insert(cumSumMaxValues, 0,0)

    maxItem = maxValues[1]
    matrixNeg = []
    output = ""

    for rowMatrix in matrix_na:
        startOutput = "1 " + str(rowMatrix[0]) + ":1 "
        endOutput = ""
        for (i, elem) in enumerate(rowMatrix[2:]):
            endOutput += str( cumSumMaxValues[i+2] + elem) + ":1 "
        for item in range( maxItem):
            output += startOutput + str(cumSumMaxValues[1] + item) + ":1 " + endOutput + " \n "
        
    with open( outputPath + '/test.libfm', 'w') as f:
        f.write(output)
    if sparse:
        command = libFMPath + "/bin/convert --ifile "
        commandOutput = outputPath + "/test.libfm --ofilex " + outputPath + "/testSparse.x --ofiley " + outputPath + "/testSparse.y"
        command = command + commandOutput

        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        p.communicate()

        command = libFMPath + "/bin/transpose --ifile " + outputPath + "/testSparse.x --ofile " + outputPath + "/testSparse.xt"

        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        p.communicate() 
    
    return
