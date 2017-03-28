import numpy as np
import scipy.sparse as sparse
import utils

class BPMF:
    def __init__(self,data,nusers,nitems,
        nanvalue=0, nfeatures=10,
        copy=True,tolerance=1.0*1.0**-3,
        seed=None,maxrating=None,minrating=None,
        maxiter=50,verbose=False,parallelized=False):

        if copy:
            self.data = data.copy()
        else:
            self.data = data

        self.nusers = nusers
        self.nitems = nitems
        self.nanvalue = nanvalue

        self.tolerance = tolerance
        self.seed = seed
        self.minrating = minrating
        self.maxrating = maxrating
        self.maxiter = maxiter

        self.verbose = verbose
        self.parallelized

        # Bayesian Probabilistic Matrix Factorization
        self.U = np.zeros((nusers,nfeatures))
        self.V = np.zeros((nitems,nfeatures))

    def train(self):
        self.meanrating = np.mean(rating[:,2])
        self.ratingmatrix = utils.record2matrix(record=self.data,nusers=self.nusers,nitems=self.nitems)
        
        lastRMSE = None
        for i in range(self.maxiter):
            self.__updateitemparams()
            self.__updateuserparams()

            self.__updateitemfeatures()
            self.__updateuserfeatures()

            preds = self.predict(self.data)
            newRMSE = utils.RMSE(preds,self.data)

            if lastRMSE and self.verbose:
                print 'RMSE of {iter}th epoch: {rmse}'.format(iter=i,rmse=newRMSE)

            if lastRMSE and np.abs(newRMSE - lastRMSE) < self.tolerance:
                print 'Converge with RMSE: {rmse}'.format(rmse=newRMSE)
                break

            lastRMSE = newRMSE
        
        if self.verbose:
            print 'Train stop. {reason}'.format(reason='Maximum Iteration!')

    def predict(self,data):
        
    
    def __updateitemparams(self):
    
    def __updateuserparams(self):
    
    def __updateitemfeatures(self):

    def __updateuserfeatures(self):