from sklearn.svm import SVC,NuSVC
import numpy as np
import matplotlib.pyplot as plt

'''
用于对XOR计算进行分类
'''

#自定义核函数
def myKernel(x,x_i):
    #print(x,x_i)
    return (1+np.dot(x,x_i.T))**2

#生成数据集和训练（对比了两种）
X=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
y=np.array([-1,1,1,-1])
#print(X,y)
svc=SVC(kernel=myKernel)
nusvc=NuSVC(kernel=myKernel)
svc.fit(X,y)
nusvc.fit(X,y)

#绘制图像（热力图）
def paint(svm,type):
    '''

    :param svm: 选择的svm分类器
    :param type:选择的svm分类器类型（用于绘图）
    :return:
    '''
    xp = np.linspace(-2, 2, 200)
    yp = np.linspace(-2, 2, 200)
    Xp, Yp = np.meshgrid(xp, yp)
    XX=Xp.copy()
    YY=Yp.copy()
    Xp.resize(40000)
    Yp.resize(40000)
    ZZ =svm.predict(np.array([Xp,Yp]).T)
    ZZ.resize(200,200)

    plt.contourf(XX, YY, ZZ, 10, alpha=0.75, cmap='RdBu')
    plt.xticks([-2,-1,0,1,2])
    plt.xlabel('x')
    plt.yticks([-2,-1,0,1,2])
    plt.ylabel('y')
    plt.title('SVM Classfier Results('+type+')')
    plt.savefig('./Image/SVMResults_'+type+'.png')
    plt.show()

#绘制图像
paint(svc,'svc')
paint(nusvc,'nusvc')

