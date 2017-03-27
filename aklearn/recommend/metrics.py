import numpy as np

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