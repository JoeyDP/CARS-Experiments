from typing import List, Union
import itertools
import subprocess

import numpy as np
import scipy.sparse
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from src.algorithm.cars.cars_algorithm import CARSAlgorithm
from src.data.cars import CARSData, CARSDataMD, CARSDataFlat, combineIndices
# from src.util import multiply, indexFactors
import src.util as util

class FM(CARSAlgorithm):
    """
    Steffen Rendle - Factorization Machines
    https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    We use the libFM implementation, see manual http://www.libfm.org/libfm-1.42.manual.pdf
    You first need to install libFM on your laptop
    """
    def __init__(self, k0: bool = True, k1: bool = True, k2: int = 10, iterations: int =10, method: str = "mcmc", nu: float = 0.01, init_stdev: float = 0.1, task: str= 'r'):
        # see http://www.libfm.org/libfm-1.42.manual.pdf for more information about the parameters
        super().__init__()
        self.k0 = int(k0)
        self.k1 = int(k1)
        self.k2 = k2
        self.iterations = iterations
        self.method = method # can have values "als", "sgd", "mcmc" or "sgda". According to ffm paper ( https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) paper, is ALS the best method.
        if not self.method in ["als", "mcmc", "sgda", "sgd"]:
            print("not a correct name for the method, we use default method mcmc")
            self.method = "mcmc"
        self.nu = nu # learning rate (for sgd)
        self.init_stdev = init_stdev
        self.task = task # r (regression) or c (classification)
        
    def setPaths( self, pathData: str, pathLibFM: str):
        """
        We set the paths where the data is stored and where the libFM executables are installed. 
        We needs this later when we apply the fit method
        """
        self.pathData = pathData
        self.pathLibFM = pathLibFM
        return self
    
    def setmn(self, data):
        self.m = data.m
        self.n = data.n
        return self
    
    def setSparse(self, sparse):
        """
        Decide whether we use the sparse implementation or not.
        """
        self.sparse = sparse 
        return self
    
    def fit(self):  
        """
        We make first the command, then we execute this command like in terminal mode 
        """
        if self.sparse:
            pathTrain = self.pathData + "/trainSparse"
            pathTest = self.pathData + "/testSparse"
        else:
            pathTrain = self.pathData + "/train.libfm"
            pathTest = self.pathData + "/test.libfm"
        pathOutput = self.pathData + "/outputFM"
        commandTaskTrain = self.pathLibFM  + "/bin/libFM" " -task " + self.task + " -train " + pathTrain + " -test " + pathTest
        
        commandMethod = " -method " + self.method
        if self.method in ["als", "sgd", "sgda"]:
            commandMethod += " -init_stdev " + str(self.init_stdev)
            if self.method == "sgd":
                commandMethod += " -learn_rate " + str(nu)
            
        
        command_hp = " -dim " + str(self.k0) + "," + str(self.k1) + "," + str(self.k2) + " -iter " + str(self.iterations)
        command_out =  pathOutput + "/fm_predict_" + str(self.k0) + "_" + str(self.k1) + "_" + str(self.k2) + "_" + str(self.iterations) + ".dat"
        command = commandTaskTrain + command_hp + " -out " + command_out
        
        # executing this command
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        p.communicate()
        
        return self

    def predict_all(self, val_user_ids: np.array, val_context_indices: List[np.array]) -> scipy.sparse.csr_matrix:
        # Checking
        
        assert val_user_ids.shape[0] == len(val_context_indices[0]), "users in and out need to correspond"
        
        # we open the file where the output is stored.
        pathOutput = self.pathData + "/outputFM"
        inputFile = pathOutput + "/fm_predict_" + str(self.k0) + "_" + str(self.k1) + "_" + str(self.k2) + "_" + str(self.iterations) + ".dat"
        with open( inputFile, "r") as f:
            predictions_help = np.array( list( map( lambda x: float(x.replace("\n", "")), f.readlines()) ))
        
        # the predictions come back in an array, this needs to be reshaped
        predictions_help = predictions_help.reshape( (len(val_user_ids), self.n) )
        predictions = np.zeros( (self.m, self.n) )
        predictions[ val_user_ids, :] = predictions_help

        return predictions
