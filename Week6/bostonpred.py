from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import random
import math
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt


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

#线性回归函数
def linear_regression(Param,X):
    '''

    :param Param: 线性回归参数
    :param X: 特征数据
    :return: 回归结果
    '''
    return np.dot(X,Param.T)

#损失函数
def mse_loss(output,target):
    '''

    :param output:回归结果
    :param target:目标结果
    :return:损失函数的值
    '''
    return np.mean((output-target)**2)

#损失函数的梯度
def grad(output,target,X):
    '''

    :param output: 回归结果
    :param target: 目标结果
    :param X: 特征数据
    :return: 损失函数的梯度
    '''
    #print(output.shape,target.shape,X.shape)
    grad_np=2*np.mean(X.T*((output-target)),axis=1)
    return grad_np.T

#损失函数hessian矩阵
def hess(X):
    '''

    :param X:特征数据，因为mse的形式缘故hessian只需要X值
    :return:hessian矩阵
    '''
    hess_np = 2*np.dot(X.T,X)/len(X)
    return hess_np

#sgd优化法
def sgd(output,target,X,Param,lr=0.001):
    '''

    :param output: 回归结果
    :param target: 目标结果
    :param X: 特征数据
    :param Param: 线性回归参数
    :param lr: 学习率
    :return:更新的回归参数
    '''
    target=target.reshape(-1,1)
    idx=random.randint(0,len(X)-1)
    Param-=lr*grad(np.array([output[idx]]),np.array([target[idx]]),np.array([X[idx]]))
    return Param

#牛顿法
def newton(output,target,X,Param,lr=0.001):
    '''

    :param output: 回归结果
    :param target: 目标结果
    :param X: 特征数据
    :param Param: 线性回归参数
    :param lr: 学习率
    :return:更新的回归参数
    '''
    g = grad(output,target,X)
    #print(g.shape)
    h = hess(X)
    if(np.linalg.det(hess(X))==0):return Param
    #print(h.shape)
    Param-=lr*np.dot(g,np.linalg.inv(h))
    return Param

#adagrad
def adagrad(output,target,X,Param,r=0,lr=0.001,d=0.0000001):
    '''

    :param output: 回归结果
    :param target: 目标结果
    :param X: 特征数据
    :param Param: 线性回归参数
    :param r:grad累计量
    :param lr:学习率
    :param d:小常数
    :return:更新的参数，更新的grad累积量
    '''
    g = grad(output,target,X)
    r+=np.sum(g*g)
    t = 1/(d+math.sqrt(r))
    Param-=lr*t*g
    return Param,r

#批量输入数据，在这里没有用
def loader(X,y,batch_size=50):
    '''

    :param X: 特征数据
    :param y: 真实标签
    :param batch_size:批量大小，用于划分批量，并不一定等于
    :return:划分完成的数据和标签
    '''
    t = len(X)/batch_size+(1 if len(X)%batch_size>0 else 0)
    X_list=np.array_split(X,t)
    y_list=np.array_split(y,t)
    #print(X_list)
    #print(y_list)
    return list(zip(X_list,y_list))



#SGD训练
def sgdtrain(lr=0.01):
    Param=Original_Param.copy()
    r=0
    x,y=X_train,y_train
    losslist=[]
    for epoch in range(EPOCH):
        out=linear_regression(Param,x)
        loss=mse_loss(out,y)
        Param=sgd(out,y,x,Param,lr=lr)
        #Param=newton(out,y,x,Param,lr=0.01)
        #Param,r=adagrad(out,y,x,Param,r,lr=0.1)
        print(epoch,':',loss)
        losslist.append(loss)
    return Param,losslist

#Newton训练
def newtontrain(lr=0.01):
    Param=Original_Param.copy()
    r=0
    x,y=X_train,y_train
    losslist=[]
    for epoch in range(EPOCH):
        out=linear_regression(Param,x)
        loss=mse_loss(out,y)
        #Param=sgd(out,y,x,Param,lr=lr)
        Param=newton(out,y,x,Param,lr=lr)
        #Param,r=adagrad(out,y,x,Param,r,lr=0.1)
        print(epoch,':',loss)
        losslist.append(loss)
    return Param,losslist

#Adagrad训练
def adagradtrain(lr=0.1):
    Param=Original_Param.copy()
    r=0
    x,y=X_train,y_train
    losslist=[]
    for epoch in range(EPOCH):
        out=linear_regression(Param,x)
        loss=mse_loss(out,y)
        #Param=sgd(out,y,x,Param,lr=lr)
        #Param=newton(out,y,x,Param,lr=lr)
        Param,r=adagrad(out,y,x,Param,r,lr=lr)
        print(epoch,':',loss)
        losslist.append(loss)
    return Param,losslist

#测试
def test(Param):
    out=linear_regression(Param,X_test)
    loss=mse_loss(out,y_test)
    #print(out.shape,y_test.shape)
    #print(loss,r2_score(y_test.reshape(-1,1),out))
    print('loss:',loss,'r2:',r2_score(y_test.reshape(-1,1),out))

#绘图
def paint(sgdloss,newtonloss,adaloss):
    if(not os.path.exists("Image/")):os.mkdir("Image/")
    plt.figure(figsize=(8,8))
    plt.subplot(3,1,1)
    plt.plot(range(EPOCH),sgdloss,c='dodgerblue')
    plt.title("SGD")
    plt.subplot(3,1,2)
    plt.plot(range(EPOCH),newtonloss,c='dodgerblue')
    plt.title("Newton")
    plt.subplot(3,1,3)
    plt.plot(range(EPOCH),adaloss,c='dodgerblue')
    plt.title("Ada")
    plt.suptitle("Loss")
    plt.savefig("Image/loss")
    plt.show()



if __name__=='__main__':
    sgdparam,sgdloss=sgdtrain()
    newtonparam,newtonloss=newtontrain()
    adagradparam,adaloss=adagradtrain()

    print('sgd results:')
    test(sgdparam)
    print('newton results:')
    test(newtonparam)
    print('adagrad results:')
    test(adagradparam)
    paint(sgdloss,newtonloss,adaloss)



