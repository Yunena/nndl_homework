from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

'''
对四种类别的SVM进行分析（对应报告2.1）
'''

#读取mnist并分配训练集测试集
mnist = fetch_openml('mnist_784',data_home='./Datasize',)
X_train,X_test,y_train,y_test=train_test_split(mnist['data'],mnist['target'],train_size=7000,test_size=3000,shuffle=True)
print('mnist have loaded.')
#打开文件及列表存储准备
filepath='Data.txt'
f=open(filepath,'w')

#存储准确率
polylist=[]
rbflist=[]
siglist=[]

#评估模型（用准确率）
def evaluate(svm):
    y_pred=svm.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    return acc


#生成并评估'linear'
def get_linearsvm():
    linearsvm=SVC(kernel='linear')
    linearsvm.fit(X_train,y_train)

    return evaluate(linearsvm)

#生成并评估'poly'
def get_polysvm(degree):#这里的超参数是degree
    #print(gamma)
    polysvm=SVC(kernel='poly',degree=degree)
    polysvm.fit(X_train,y_train)
    return evaluate(polysvm)

#生成并评估'rbf'
def get_rbfsvm(gamma):#超参数为gamma
    #print(gamma)
    rbfsvm=SVC(kernel='rbf',gamma=gamma)
    rbfsvm.fit(X_train,y_train)
    return evaluate(rbfsvm)

#生成并评估'sigmoid'
def get_sigsvm(coef0):#超参数为coef0
    sigsvm=SVC(kernel='sigmoid',coef0=coef0)
    sigsvm.fit(X_train,y_train)
    return evaluate(sigsvm)

#绘图
def type_paint(times):
    '''

    :param times: 循环次数（取值上限，报告中是4）
    :return:
    '''
    plt.figure(figsize=(6,10))
    x=list(range(times))
    plt.subplot(311)
    plt.plot(x,polylist,c='b')
    plt.ylabel('acc')
    plt.title('poly(degree)')
    plt.subplot(312)
    plt.plot(x,rbflist,c='b')
    plt.ylabel('acc')
    plt.title('rbf(coef0)')
    plt.subplot(313)
    plt.plot(x,siglist,c='b')
    plt.ylabel('acc')
    plt.title('sigmoid(coef0)')
    plt.savefig('./Image/type_paint.png')
    plt.show()



#开始运行
Times=5
acc=get_linearsvm()
f.writelines('linear:'+str(acc))
print('linear:',get_linearsvm())

print('poly:')
for i in range(Times):
    acc=get_polysvm(i)
    polylist.append(acc)
    print(acc,end=' ')
f.writelines('poly:'+str(polylist))
print('\nrbf:')
for i in range(Times):
    acc=get_rbfsvm(float(0.1**i))
    rbflist.append(acc)
    print(acc,end=' ')
f.writelines('rbf:'+str(rbflist))
print('\nsigmoid:')
for i in range(Times):
    acc = get_sigsvm(float(0.1**i))
    siglist.append(acc)
    print(acc,end=' ')
f.writelines('sigmoid:'+str(siglist))
f.close()
type_paint(Times)

