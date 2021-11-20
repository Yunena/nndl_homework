from sklearn.neural_network import MLPClassifier
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
对MLP超参数进行分析
'''

if not os.path.exists('Image/'): os.mkdir('Image/')

start = time.time()
mnist = fetch_openml('mnist_784',data_home='./Datasize')
print('mnist have loaded.',time.time() - start)#显示load时间
data=mnist['data']
target=mnist['target']
X_train,X_test,y_train,y_test = train_test_split(mnist['data'],mnist['target'],train_size = 7000,test_size = 3000, shuffle = True)

#训练并测试fc
def train_and_test_fc(max_iter=200, lr = 0.001):
    fc = MLPClassifier(max_iter=max_iter, learning_rate_init=lr)
    start = time.time()
    fc.fit(X_train,y_train)
    train_time = time.time() - start
    y_pred = fc.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    return train_time, acc

#对迭代次数进行分析
def iteration_analysis():
    ite_list = range(10,201)
    time_list = []
    acc_list = []
    for ite in ite_list:
        train_time, acc = train_and_test_fc(max_iter=ite)
        time_list.append(train_time)
        acc_list.append(acc)
        print('Now:', ite, 'Time costs:',train_time, 'Acc:',acc)

    plt.plot(ite_list,acc_list,c='g')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Trends')
    plt.savefig('Image/fc_accuracy_list')
    plt.show()

    plt.plot(ite_list, time_list, c='b')
    plt.xlabel('Iteration')
    plt.ylabel('Time cost')
    plt.title('Time Trends')
    plt.savefig('Image/fc_time_list')
    plt.show()

#对学习率进行分析
def lr_analysis():
    lr_list = [0.1**i for i in range(1,11)]
    acc_list = []
    for lr in lr_list:
        _, acc = train_and_test_fc(lr=lr)
        acc_list.append(acc)

    plt.plot(range(1,11), acc_list, c='g')
    plt.xlabel('Power')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Trends')
    plt.savefig('Image/fc_lr_accuracy_list')
    plt.show()




