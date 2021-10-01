import numpy as np
from functions import *


'''
grad_des:梯度下降法实现
'''

def grad_des(epoch=100,lr_list=[1,0.1,0.01,0.001,0.0001,0.00001]):
    '''
    epoch:迭代次数，默认为100
    lr_list:学习率可选范围，由函数自己选择其中合适的学习率
    return: 迭代过程中的位置变化（自变量向量X和目标值y）
    '''
    init_X=np.random.rand(2)
    print('Init:',init_X)
    lr_np = np.array(lr_list)
    lr_len = len(lr_list)
    X_list=[]
    y_list=[]

    X = init_X
    X_list.append(X)
    y_list.append(f(X[0],X[1]))
    print('Current Spot:',X,'Current Height:',f(X[0],X[1]))
    for i in range(epoch):
        grad_X = grad(X[0],X[1])
        lr = lr_list[np.argmin(f(X[0]*np.ones(lr_len)-lr_np*grad_X[0],X[1]*np.ones(lr_len)-lr_np*grad_X[1]))]
        X = X-lr*grad_X
        y = f(X[0],X[1])
        print('Current LR:',lr,'Current Spot:',X,'Current Height:',y)
        if y==float('inf') or y==-float('inf'):
            break
        X_list.append(X)
        y_list.append(y)
    return np.array(X_list).T,np.array(y_list)


#grad_des()


