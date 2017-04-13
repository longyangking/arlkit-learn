import sys
sys.path.append("..")

import aklearn.recommend as RE
import numpy as np

def generatedata(m,n):
    ratings = np.zeros((m*n,3))
    eps = 0.1
    for i in range(m):
        for j in range(n):
            if np.random.random() < eps:
                ratings[i*n+j] = [i,j,np.random.randint(6)]
    return ratings

if __name__ == '__main__':
    nusers,nitems = 100,70
    ratings = generatedata(nusers,nitems)
    trainratings,testratings = RE.utils.cv(ratings)
    traindata = RE.utils.record2matrix(trainratings,nusers,nitems)
    testdata = RE.utils.record2matrix(testratings,nusers,nitems)

    re = RE.SCR(traindata,selfconsistence=False,verbose=True)
    re.train()
    userpred,itempred = re.predict(traindata)

    print 'Without SCR, User-based RMSE: {rmse}'.format(rmse=RE.utils.RMSE(testdata,userpred))          
    print 'Without SCR, Item-based RMSE: {rmse}'.format(rmse=RE.utils.RMSE(testdata,itempred))

    re = RE.SCR(traindata,selfconsistence=True)
    re.train()
    userpred,itempred = re.predict(traindata)

    print 'With SCR, User-based RMSE: {rmse}'.format(rmse=RE.utils.RMSE(testdata,userpred))          
    print 'With SCR, Item-based RMSE: {rmse}'.format(rmse=RE.utils.RMSE(testdata,itempred))
