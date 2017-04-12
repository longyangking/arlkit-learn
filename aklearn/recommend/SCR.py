import numpy as np
import pandas as pd
import utils

class SCR:
    def __init__(self,matrix,format='cosine',nanvalue=0,copy=True,\
            selfconsistence=True,tolerance=1.0*1.0**-6,verbose=False):
        if copy:
            self.trainmatrix = matrix.copy()
        else:
            self.trainmatrix = matrix

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
        self.verbose = verbose

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
        for i in range(self.nusers):
            for j in range(i+1,self.nusers):
                self.usersimilarity[i,j] = self.cosinesimilarity(self.trainmatrix[i,:],self.trainmatrix[j,:])
        self.usersimilarity = self.usersimilarity + self.usersimilarity.transpose()
        for i in range(self.nusers):
            self.usersimilarity[i,i] = self.cosinesimilarity(self.trainmatrix[i,:],self.trainmatrix[i,:])

        for i in range(self.nitems):
            for j in range(i+1,self.nitems):
                self.itemsimilarity[i,j] = self.cosinesimilarity(self.trainmatrix[:,i],self.trainmatrix[:,j])
        self.itemsimilarity = self.itemsimilarity + self.itemsimilarity.transpose()
        for i in range(self.nitems):
            self.itemsimilarity[i,i] = self.cosinesimilarity(self.trainmatrix[:,i],self.trainmatrix[:,i])

        if self.verbose:
            print 'Train Complete!'

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

        drmse = None
        lastrmse = None
        userdata = data
        index = 0

        while (drmse is None) or (drmse > self.tolerance):          
            meanusermatrix = np.mean(userdata,axis=1)
            usermatrix = userdata - meanusermatrix[:,np.newaxis]

            # Singular point
            userpred = meanusermatrix[:,np.newaxis]\
                + self.usersimilarity.dot(usermatrix)/np.array([np.abs(self.usersimilarity).sum(axis=1)]).T

            if not self.selfconsistence:
                break
            
            if lastrmse is None:
                lastrmse = utils.RMSE(userdata,userpred)
                userdata = userpred
                continue

            index += 1
            currentrmse = utils.RMSE(userdata,userpred)
            drmse = lastrmse - currentrmse
            lastrmse = currentrmse
            userdata = userpred

            if self.verbose:
                print '{index}th Iteration -> SCR User-Prediction with RMSE: {rmse}'.format(index=index,rmse=currentrmse)
            
        drmse = None
        lastrmse = None  
        itemdata = data
        index = 0
        while (drmse is None) or (drmse > self.tolerance):
    
            itempred = itemdata.dot(self.itemsimilarity)/np.array([np.abs(self.itemsimilarity).sum(axis=1)])

            if not self.selfconsistence:
                break

            if lastrmse is None:
                lastrmse = utils.RMSE(itemdata,itempred)
                itemdata = itempred
                continue
            
            index += 1
            currentrmse =  utils.RMSE(itemdata,itempred)
            drmse = lastrmse - currentrmse
            lastrmse = currentrmse
            itemdata = itempred


            if self.verbose:
                print '{index}th Iteration -> SCR Item-Prediction with RMSE: {rmse}'.format(index=index,rmse=currentrmse)

        return userpred,itempred
