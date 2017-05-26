import sys
sys.path.append("..")

import aklearn.recommend as RE
import numpy as np

def generatedata(m,n):
    ratings = list()
    eps = 0.1
    for i in range(m):
        for j in range(n):
            if np.random.random() < eps:
                ratings.append([i,j,np.random.randint(1,6)])
    return np.array(ratings)

if __name__ == '__main__':
    nusers,nitems = 100,70
    ratings = generatedata(nusers,nitems)
    trainratings,testratings = RE.utils.cv(ratings)
    #traindata = RE.utils.record2matrix(trainratings,nusers,nitems)
    
    bpmf = RE.BPMF(nusers,nitems,maxrating=5,minrating=1,verbose=True)

    bpmf.train(trainratings)

    preds = bpmf.predict(trainratings)

    testdata = RE.utils.record2matrix(testratings,nusers,nitems)
    preds = np.column_stack((trainratings[:,0],trainratings[:,1],preds))
    predsdata = RE.utils.record2matrix(preds,nusers,nitems)


    print 'BPMF Test-RMSE: {rmse}'.format(rmse=RE.utils.RMSE(testdata,predsdata))          