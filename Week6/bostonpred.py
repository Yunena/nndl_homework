from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import random
import math
from sklearn.preprocessing import StandardScaler

#导入数据集
bostons = load_boston()
X=bostons['data']
Y=bostons['target']
input_num=X.shape[1]
X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.7)
ssx=StandardScaler()
ssy=StandardScaler()
X_train=ssx.fit_transform(X_train)
X_test=ssx.transform(X_test)
y_train=ssy.fit_transform(y_train.reshape(-1,1)).reshape(1,-1)
y_test=ssy.transform(y_test.reshape(-1,1)).reshape(1,-1)
b_train = np.ones(len(X_train))
b_test = np.ones(len(X_test))
X_train = np.c_[X_train,b_train]
X_test = np.c_[X_test,b_test]
Original_Param = np.random.randn(input_num+1)




EPOCH=10000

def linear_regression(Param,X):
    return np.dot(X,Param.T)

def mse_loss(output,target):
    return np.mean((output-target)**2)

def grad(output,target,X):
    #print(output.shape,target.shape,X.shape)
    grad_np=2*np.mean(X.T*((output-target)),axis=1)
    return grad_np.T

def hess(X):
    hess_np = 2*np.dot(X.T,X)/len(X)
    return hess_np

def sgd(output,target,X,Param,lr=0.001):
    target=target.reshape(-1,1)
    idx=random.randint(0,len(X)-1)
    Param-=lr*grad(np.array([output[idx]]),np.array([target[idx]]),np.array([X[idx]]))
    return Param

def newton(output,target,X,Param,lr=0.001):
    g = grad(output,target,X)
    #print(g.shape)
    h = hess(X)
    if(np.linalg.det(hess(X))==0):return Param
    #print(h.shape)
    Param-=lr*np.dot(g,np.linalg.inv(h))
    return Param

def adagrad(output,target,X,Param,r=0,lr=0.001,d=0.0000001):
    g = grad(output,target,X)
    r+=np.sum(g*g)
    t = 1/(d+math.sqrt(r))
    Param-=lr*t*g
    return Param,r

def loader(X,y,batch_size=50):
    t = len(X)/batch_size+(1 if len(X)%batch_size>0 else 0)
    X_list=np.array_split(X,t)
    y_list=np.array_split(y,t)
    #print(X_list)
    #print(y_list)
    return list(zip(X_list,y_list))




def sgdtrain(lr=0.01):
    Param=Original_Param.copy()
    r=0
    x,y=X_train,y_train
    for epoch in range(EPOCH):
        out=linear_regression(Param,x)
        loss=mse_loss(out,y)
        Param=sgd(out,y,x,Param,lr=lr)
        #Param=newton(out,y,x,Param,lr=0.01)
        #Param,r=adagrad(out,y,x,Param,r,lr=0.1)
        print(epoch,':',loss)
    return Param

def newtontrain(lr=0.01):
    Param=Original_Param.copy()
    r=0
    x,y=X_train,y_train
    for epoch in range(EPOCH):
        out=linear_regression(Param,x)
        loss=mse_loss(out,y)
        #Param=sgd(out,y,x,Param,lr=lr)
        Param=newton(out,y,x,Param,lr=lr)
        #Param,r=adagrad(out,y,x,Param,r,lr=0.1)
        print(epoch,':',loss)
    return Param

def adagradtrain(lr=0.1):
    Param=Original_Param.copy()
    r=0
    x,y=X_train,y_train
    for epoch in range(EPOCH):
        out=linear_regression(Param,x)
        loss=mse_loss(out,y)
        #Param=sgd(out,y,x,Param,lr=lr)
        #Param=newton(out,y,x,Param,lr=lr)
        Param,r=adagrad(out,y,x,Param,r,lr=lr)
        print(epoch,':',loss)
    return Param


def test(Param):
    out=linear_regression(Param,X_test)
    loss=mse_loss(out,y_test)
    #print(out.shape,y_test.shape)
    #print(loss,r2_score(y_test.reshape(-1,1),out))
    print('loss:',loss,'r2:',r2_score(y_test.reshape(-1,1),out))

if __name__=='__main__':
    sgdparam=sgdtrain()
    newtonparam=newtontrain()
    adagradparam=adagradtrain()

    print('sgd results:')
    test(sgdparam)
    print('newton results:')
    test(newtonparam)
    print('adagrad results:')
    test(adagradparam)



