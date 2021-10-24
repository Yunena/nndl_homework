from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
import time

'''
对四类SVM，在其表现最好的参数下分别比较其时间
'''

#读取数据集
mnist = fetch_openml('mnist_784',data_home='./Datasize',)
X_train,X_test,y_train,y_test=train_test_split(mnist['data'],mnist['target'],train_size=7000,test_size=3000,shuffle=True)
print('mnist have loaded.')

#生成模型
linearsvm=SVC(kernel='linear')
polysvm=SVC(kernel='poly',degree=2.0)#degree为2的时候能表现出极值
rbfsvm=SVC(kernel='rbf')
sigsvm=SVC(kernel='sigmoid',coef0=0.01)#coef0为0.01的时候能表现出极值

train_time=[]

#训练
start=time.time()
linearsvm.fit(X_train,y_train)
train_time.append(time.time()-start)
start=time.time()
polysvm.fit(X_train,y_train)
train_time.append(time.time()-start)
start=time.time()
rbfsvm.fit(X_train,y_train)
train_time.append(time.time()-start)
start=time.time()
sigsvm.fit(X_train,y_train)
train_time.append(time.time()-start)
print('training has finished.')

test_time=[]
#测试
start=time.time()
linear_pred=linearsvm.predict(X_test)
test_time.append(time.time()-start)
start=time.time()
poly_pred=polysvm.predict(X_test)
test_time.append(time.time()-start)
start=time.time()
rbf_pred=rbfsvm.predict(X_test)
test_time.append(time.time()-start)
start=time.time()
sig_pred=sigsvm.predict(X_test)
test_time.append(time.time()-start)

#计算准确率
accuracy=[]
accuracy.append(accuracy_score(y_test,linear_pred))
accuracy.append(accuracy_score(y_test,poly_pred))
accuracy.append(accuracy_score(y_test,rbf_pred))
accuracy.append(accuracy_score(y_test,sig_pred))

print('train time:',train_time)
print('test time:',test_time)
print('accuracy:',accuracy)
