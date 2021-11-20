from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random

'''
比较svm和fc
'''

if not os.path.exists('Image/'): os.mkdir('Image/')

start = time.time()
mnist = fetch_openml('mnist_784',data_home='./Datasize')
print('mnist have loaded.',time.time() - start)#显示load时间
data=mnist['data']
target=mnist['target']
X_train,X_test,y_train,y_test = train_test_split(mnist['data'],mnist['target'],train_size = 0.7, shuffle = True)

def train_and_test_fc(max_iter=200, lr = 0.001):
    fc = MLPClassifier(max_iter=max_iter, learning_rate_init=lr)
    start = time.time()
    fc.fit(X_train,y_train)
    train_time = time.time() - start
    start = time.time()
    y_pred = fc.predict(X_test)
    test_time = time.time()-start
    acc = accuracy_score(y_test,y_pred)
    return train_time, test_time, acc

def train_and_test_svm():
    svm = SVC(kernel='rbf')
    start = time.time()
    svm.fit(X_train,y_train)
    train_time = time.time() - start
    start = time.time()
    y_pred = svm.predict(X_test)
    test_time = time.time()-start
    acc = accuracy_score(y_test,y_pred)
    return train_time, test_time, acc

def compare():
    print('FC:')
    train_time,test_time,acc=train_and_test_fc()
    print('train_time:',train_time,'test_time:',test_time,'acc:',acc)
    print('SVM:')
    train_time,test_time,acc=train_and_test_svm()
    print('train_time:',train_time,'test_time:',test_time,'acc:',acc)


if __name__=="__main__":
    compare()
