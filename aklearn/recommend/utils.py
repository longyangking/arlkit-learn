import numpy as np
import scipy.sparse as sparse
import random

def record2matrix(record,nusers,nitems):
    if record[0,0] == 1 and record[0,1] == 1:
        record[:,(0,1)] = 0

    #data = record[:,2]
    #rowindex = record[:,0]
    #colindex = record[:,1]
    #shape = (nusers,nitems)
    #return sparse.csr_matrix((data,(rowindex,colindex)),shape=shape)

    matrix = np.zeros((nusers,nitems))
    pos = (record[:,0].astype(int),record[:,1].astype(int))
    matrix[pos] = record[:,2] 
    return matrix

def cv(record,ratio=0.1):
    (M,N) = record.shape
    sample = np.array(random.sample(range(M),int(M*ratio)))
    test = record[sample]
    trainsample = np.array([x for x in range(M) if x not in sample])
    train = record[trainsample]
    return train,test

def RMSE(prediction,truth,pos=None):
    '''
    Calculate Root Mean Square Error (RMSE)
    '''
    if pos is None:
        pos = np.where(truth != 0)
    
    filterpred = prediction[pos]
    filtertruth = truth[pos]
    
    error = np.sqrt(np.sum(np.square(np.abs(filterpred-filtertruth))))
    return error