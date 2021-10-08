import numpy as np

'''
standardstep:对矩阵进行标准化
'''

def standardstep(X):
    '''

    :param X: 准备标准化的矩阵
    :return: 被标准化的矩阵
    '''
    #print(X.shape)
    mean = np.mean(X,axis=0)
    std = np.std(X,ddof=1,axis=0)
    print(X,std)
    X_T = X.T
    for i in range(len(X_T)):
        X_T[i]-=mean[i]*np.ones(len(X))
        if(std[i]!=0):
            X_T[i]/=std[i]
    return X_T.T