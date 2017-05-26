import numpy as np
import scipy.sparse as sparse
from numpy.linalg import inv,cholesky
from numpy.random import RandomState
from scipy.stats import wishart
import utils

class BPMF:
    '''
    Bayesian Probabilistic Matrix Factorization
    '''
    def __init__(self,nusers,nitems,nfeatures=10,
        beta=2.0,betauser=2.0,dfuser=None,betaitem=2.0,dfitem=None,
        tolerance=1.0*1.0**-6,nanvalue=0,
        seed=None,maxrating=None,minrating=None,
        verbose=False,parallelized=False):

        self.nusers = nusers
        self.nitems = nitems
        self.nfeatures = nfeatures

        self.randstate = RandomState(seed) if seed is not None else RandomState(0)
        self.nanvalue = nanvalue

        self.tolerance = tolerance
        self.minrating = minrating
        self.maxrating = maxrating

        self.verbose = verbose
        self.parallelized = parallelized

        # Bayesian Probabilistic Matrix Factorization
        self.beta = beta

        self.WIuser = np.eye(self.nfeatures,dtype='float64')
        self.alphauser = np.zeros((self.nfeatures,1),dtype='float64')
        self.betauser = betauser
        self.dfuser = int(dfuser) if dfuser is not None else self.nfeatures
        self.muuser = np.zeros((self.nfeatures,1),dtype='float64')

        self.WIitem = np.eye(self.nfeatures,dtype='float64')
        self.alphaitem = np.zeros((self.nfeatures,1),dtype='float64')
        self.betaitem = betaitem
        self.dfitem = int(dfitem) if dfitem is not None else self.nfeatures
        self.muitem = np.zeros((self.nfeatures,1),dtype='float64')

        self.userfeatures = np.zeros((nusers,nfeatures))
        self.itemfeatures = np.zeros((nitems,nfeatures))

        self.__meanrating = None
        self.__ratings_csr = None
        self.__ratings_csc = None

    def train(self,ratings,maxiter=50):
        self.__meanrating = np.mean(ratings[:,2])

        # TODO this shall be sparse matrix
        self.__ratings_csr = utils.record2matrix(record=ratings,nusers=self.nusers,nitems=self.nitems)
        self.__ratings_csc = self.__ratings_csr.tocsc()
        
        lastRMSE = None
        for i in range(maxiter):
            self.__updateitemparams()
            self.__updateuserparams()

            self.__updateitemfeatures()
            self.__updateuserfeatures()

            # Compute RMSE
            preds = self.predict(ratings[:,:2])
            newRMSE = utils.RMSE(preds,ratings[:,2])

            if lastRMSE and self.verbose:
                print 'RMSE of {iter}th epoch: {rmse}'.format(iter=i,rmse=newRMSE)
                lastRMSE = newRMSE
                continue

            if lastRMSE and np.abs(newRMSE - lastRMSE) < self.tolerance:
                print 'Converge with RMSE: {rmse}'.format(rmse=newRMSE)
                break

            lastRMSE = newRMSE
        
        else:
            if self.verbose:
                print 'Train stop. {reason}'.format(reason='Maximum Iteration!')

        # I always think the following code is amazing
        return self

    def predict(self,data):
        if not self.__meanrating:
            raise Exception("Error: Not Fit before")

        userfeatures = self.userfeatures[data[:,0],:]
        itemfeatures = self.itemfeatures[data[:,1],:]
        preds = np.sum(userfeatures*itemfeatures,axis=1) + self.__meanrating

        if self.maxrating:
            preds[preds > self.maxrating] = self.maxrating
        
        if self.minrating:
            preds[preds < self.minrating] = self.minrating
        
        return preds

    def __updateitemparams(self):
        N = self.nitems
        Xbar = np.mean(self.itemfeatures,axis=0)
        Xbar = np.reshape(Xbar,(self.nfeatures,1))
        Sbar = np.cov(self.itemfeatures.T)
        normXbar = Xbar - self.muitem

        # Update Alpha of item
        WIpost = inv(inv(self.WIitem) + N*Sbar + np.dot(normXbar,normXbar.T)*(N*self.betaitem)/(self.betaitem + N))
        WIpost = (WIpost + WIpost.T)/2.0
        dfpost = self.dfitem + N
        self.alphaitem = wishart.rvs(dfpost, WIpost, 1, self.randstate)

        # Update Mu of item
        mutemp = (self.betaitem*self.muitem + N*Xbar)/(self.betaitem + N)
        lam = cholesky(inv(np.dot(self.betaitem + N, self.alphaitem)))
        self.muitem = mutemp + np.dot(lam, self.randstate.randn(self.nfeatures,1))
    
    def __updateuserparams(self):
        N = self.nusers
        Xbar = np.mean(self.userfeatures,0).T
        Xbar = np.reshape(Xbar,(self.nfeatures,1))
        Sbar = np.cov(self.userfeatures.T)
        normXbar = Xbar - self.muuser

        # Update Alpha of user
        WIpost = inv(inv(self.WIuser) + N*Sbar + np.dot(normXbar,normXbar.T)*(N*self.betauser)/(self.betauser + N))
        WIpost = (WIpost + WIpost.T)/2.0
        dfpost =  self.dfuser + N
        self.alphauser = wishart.rvs(dfpost, WIpost, 1, self.randstate)

        # Update Mu of user
        mutemp = (self.betauser*self.muuser + N*Xbar)/(self.betauser + N)
        lam = cholesky(inv(np.dot(self.betauser + N, self.alphauser)))
        self.muuser = mutemp + np.dot(lam, self.randstate.randn(self.nfeatures,1))

    def __updateitemfeatures(self):
        for itemid in range(self.nitems):
            indices = self.__ratings_csc[:,itemid].indices
            features = self.userfeatures[indices,:]
            rating = self.__ratings_csc[:,itemid].data - self.__meanrating
            rating = np.reshape(rating, (rating.shape[0],1))

            covar = inv(self.alphaitem + self.beta*np.dot(features.T,features))
            lam = cholesky(covar)
            temp = self.beta*np.dot(features.T,rating) + np.dot(self.alphaitem,self.muitem)
            mean = np.dot(covar, temp)

            tempfeature = mean + np.dot(lam, self.randstate.randn(self.nfeatures,1))
            tempfeature = np.reshape(tempfeature, (self.nfeatures,))
            self.itemfeatures[itemid,:] = tempfeature

    def __updateuserfeatures(self):
        for userid in range(self.nusers):
            indices = self.__ratings_csr[userid,:].indices
            features = self.itemfeatures[indices,:]
            rating = self.__ratings_csc[userid,:].data - self.__meanrating
            rating = np.reshape(rating, (rating.shape[0],1))

            covar = inv(self.alphauser + self.beta*np.dot(features.T,features))
            lam = cholesky(covar)
            temp = self.beta*np.dot(features.T,rating) + np.dot(self.alphauser,self.muuser)
            mean = np.dot(covar, temp)

            tempfeature = mean + np.dot(lam, self.randstate.randn(self.nfeatures,1))
            tempfeature = np.reshape(tempfeature,(self.nfeatures,))
            self.userfeatures[userid,:] = tempfeature