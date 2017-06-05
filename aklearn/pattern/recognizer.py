import numpy as np
import distribution

class recognizer:
    def __init__(self,dataset,verbose=False):
        self.dataset = np.array(dataset)
        shape = self.dataset.shape
        
        self.dimensionsize = len(shape)
        self.dimensions = shape
        
        self.model = None
        self.error = None
        self.verbose = verbose
    
    def recognize(self):
        # To fit the distribution by using least square method
        distlist = distribution.distlist()
        distlistnames = dislist.names()
        errors = np.zeros(len(distlistnames))
        fitlist = list()
        for i in range(len(dislistnames)):
            name = dislistnames[i]
            dist = dislist.dist(name)
            errors[i] = dist.fit(self.dataset)
            fitlist.append(dist)
        
        minindex = np.argmin(errors)        
        self.error = np.min(errors)
        self.model = fitlist[minindex]

    def datainfo(self):
        # Return: 
        # The distribution that fit data best
        # Core parameters of distribution
        # Reliability and Quality
        error = self.error
        model = self.model
        return model,error
