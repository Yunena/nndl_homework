from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
针对'poly'进行细化的测试分析
'''

#读取mnist并分配训练集测试集
mnist = fetch_openml('mnist_784',data_home='./Datasize',)
X_train,X_test,y_train,y_test=train_test_split(mnist['data'],mnist['target'],train_size=1200,test_size=500,shuffle=True)
print('mnist have loaded.')
#print(pd.value_counts(y_test))

#degree,coef0
#生成degree和coef0，为画图做准备

degree = np.arange(1,3,0.1)
coef0 = np.arange(0,0.1,0.01)
xlen=len(degree)
ylen=len(coef0)

X,Y=np.meshgrid(degree,coef0)
x=X.copy()
y=Y.copy()
x.resize(xlen*ylen)
y.resize(xlen*ylen)
z=[]

#开始生成svm并测试
for i in range(xlen*ylen):
    #print(i,x[i],y[i])
    svm=SVC(kernel='poly',degree=x[i],coef0=y[i])
    svm.fit(X_train,y_train)
    y_pred=svm.predict(X_test)
    z.append(accuracy_score(y_test,y_pred))

print(z)

#绘图
Z=np.array(z)
Z.resize(ylen,xlen)

plt.contourf(X,Y,Z,cmap='RdBu_r')

plt.xlabel('degree')
plt.ylabel('coef0')
plt.savefig('Image/poly_paint.png')
plt.show()
