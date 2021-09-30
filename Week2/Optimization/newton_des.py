import numpy as np
from functions import *

def newton_des(epoch=100):
    init_X=np.random.rand(2)
    print('Init:',init_X)
    X_list=[]
    y_list=[]

    X = init_X
    #X=np.array([2,2])
    X_list.append(X)
    y_list.append(f(X[0],X[1]))
    print('Current Spot:',X,'Current Height:',f(X[0],X[1]))

    for i in range(epoch):
        #print(hessian(X[0],X[1]))
        gra = grad(X[0],X[1])
        hes = hessian(X[0],X[1])
        X = X-np.dot(grad(X[0],X[1]),np.linalg.inv(hessian(X[0],X[1])))
        y = f(X[0],X[1])
        X_list.append(X)
        y_list.append(y)

        if(grad(X[0],X[1]).all()==0. and np.linalg.det(hessian(X[0],X[1]))<0.): #防止鞍点
            X = X-0.0001*np.sign(gra)


        print('Current Spot:',X,'Current Height:',y)

    return np.array(X_list).T, np.array(y_list)


#newton_des()


