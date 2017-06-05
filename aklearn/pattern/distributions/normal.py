import numpy as np

class normal:
    def __init__(self,dataset):
        self.dataset = dataset
        self.samplesize = np.size(dataset,0)
        self.featuresize = np.size(dataset,1)
    
        self.meanvector = None
        self.covariancematrix = None
   
    def fit(self):
        self.meanvector = np.mean(self.dataset,axis=0)
        self.covariancematrix = np.cov(self.dataset)
        
        
