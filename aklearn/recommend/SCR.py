import numpy as np
import pandas as pd

class SCR:
    def __init__(self,matrix,format='cosine',nanvalue=0,copy=True,\
            selfconsistence=True,tolerance=1.0*1.0**-3):
        if copy:
            self.trainmatrix = trainmatrix.copy()
        else:
            self.trainmatrix = trainmatrix

        self.copy = copy
        self.format = format
        (Nusers,Nitems) = self.trainmatrix.shape
        self.nusers = Nusers
        self.nitems = Nitems

        self.usersimilarity = np.zeros((Nusers,Nusers))
        self.itemsimilarity = np.zeros((Nitems,Nitems))

        self.nanvalue = nanvalue

        # Self-consistent filtering
        self.selfconsistence = selfconsistence
        self.tolerance = tolerance

        self.init()

    def init(self):
        '''
        Deal with the value 'NaN' in the matrix
        '''
        nanpos = np.where(self.trainmatrix!=self.trainmatrix)
        self.trainmatrix[nanpos] = self.nanvalue

    def cosinesimilarity(self,A,B):
        '''
        Calculate the similarity between two vector A and B
        '''
        normA = np.sqrt(A.dot(A))
        normB = np.sqrt(B.dot(B))
        cosine = A.dot(B)
        return cosine/normA/normB

    def train(self):
        '''
        Train Similarity Matrix (User-User and Item-Item)
        '''
        for i in range(Nusers):
            for j in range(i+1,Nusers):
                self.usersimilarity[i,j] = self.cosinesimilarity(self.trainmatrix[i,:],self.trainmatrix[j,:])
        self.usersimilarity = self.usersimilarity + self.usersimilarity.transpose()
        for i in range(Nusers):
            self.usersimilarity[i,i] = self.cosinesimilarity(self.trainmatrix[i,:],self.trainmatrix[i,:])

        for i in range(Nitems):
            for j in range(i+1,Nitems):
                self.itemsimilarity[i,j] = self.cosinesimilarity(self.trainmatrix[:,i],self.trainmatrix[:,j])
        self.itemsimilarity = self.itemsimilarity + self.itemsimilarity.transpose()
        for i in range(Nitems):
            self.itemsimilarity[i,i] = self.cosinesimilarity(self.trainmatrix[:,i],self.trainmatrix[:,i])

    def predict(self,inputmatrix):
        '''
        Predict the recommendation values based on the user-user and item-item similarity
        '''
        if self.copy:
            data = inputmatrix.copy()
        else:
            data = inputmatrix

        nanpos = np.where(data!=data)
        data[nanpos] = self.nanvalue

        meanusermatrix = np.mean(data,axis=1)
        usermatrix = data - meanusermatrix[:,np.newaxis]

        # Singular point
        userpred = meanusermatrix[:,np.newaxis]\
            + usersimilarity.dot(usermatrix)/np.array([np.abs(usersimilarity).sum(axis=1)]).T
        
        itempred = data.dot(itemsimilarity)/np.array([np.abs(itemsimilarity).sum(axis=1)])
        return userpred,itempred
