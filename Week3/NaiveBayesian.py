from sklearn.naive_bayes import *
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

'''
nb_diff():用GaussianNB和BernoulliNB两个分类器，表示不同分类器对同一个数据集(make_moons)的差异，另外两个无法处理负值故不参与比较。
data_diff():用GaussianNB对不同数据集分类的结果。
paint_data_diff(Tex,Tey,y,type,acc):绘制对data_diff()不同的图。
nb_paint():对不同分类器绘图。
'''


gaus = GaussianNB()
bern = BernoulliNB()
muti = MultinomialNB()
comp = ComplementNB()

moons = make_moons(1000)
circles = make_circles(1000)
classfication = make_classification(1000)






def nb_diff():
    Trx,Tex,Try,Tey = train_test_split(moons[0],moons[1],train_size=0.7,shuffle=True)
    nb_paint(Tex,Tey,type='Moon Target')
    #plt.title('Moon Target')
    plt.savefig('./Image/moon_test_target.png')
    plt.show()

    gaus.fit(Trx,Try)
    bern.fit(Trx,Try)

    gaus_y = gaus.predict(Tex)
    bern_y = bern.predict(Tex)
    fig = plt.figure()

    plt.suptitle('Predicted Results')
    plt.axis('off')

    ax1= fig.add_subplot(121)
    nb_paint(Tex,gaus_y,ax1,'Gaussian')
    ax2= fig.add_subplot(122)
    nb_paint(Tex,bern_y,ax2,'Bernoulli')
    plt.savefig('./Image/moon_results.png')
    plt.show()

def data_diff():
    Trx, Tex, Try, Tey = train_test_split(moons[0], moons[1], train_size=0.7,shuffle=True)
    gaus = GaussianNB()
    gaus.fit(Trx,Try)
    y=gaus.predict(Tex)
    acc = accuracy_score(Tey,y)
    paint_data_diff(Tex,Tey,y,'Moons',acc)
    print(acc)

    Trx, Tex, Try, Tey = train_test_split(circles[0], circles[1], train_size=0.7,shuffle=True)
    gaus = GaussianNB()
    gaus.fit(Trx, Try)
    y = gaus.predict(Tex)
    acc = accuracy_score(Tey,y)
    paint_data_diff(Tex, Tey, y, 'Circles',acc)
    print(acc)

    Trx, Tex, Try, Tey = train_test_split(classfication[0], classfication[1], train_size=0.7,shuffle=True)
    gaus = GaussianNB()
    gaus.fit(Trx, Try)
    y = gaus.predict(Tex)
    acc = accuracy_score(Tey,y)
    paint_data_diff(Tex, Tey, y, 'Classfication',acc)
    print(acc)

def paint_data_diff(Tex,Tey,y,type,acc):
    '''

    :param Tex: 测试集数据
    :param Tey: 测试集标签
    :param y: 预测的标签结果
    :param type: 数据集类型，用以输出图像title
    :param acc: 准确率，用以输出图像title
    :return: 无
    '''
    fig = plt.figure()

    plt.suptitle(type+' results'+' acc= '+str(round(acc,2)))
    plt.axis('off')

    ax1= fig.add_subplot(121)
    nb_paint(Tex,Tey,ax1,'Targets')
    ax2= fig.add_subplot(122)
    nb_paint(Tex,y,ax2,'Results')
    plt.savefig('./Image/'+type+'.png')
    plt.show()

def nb_paint(Tex,Tey,ax=None,type='gaus'):
    '''

    :param Tex: 测试集数据
    :param Tey: 测试集标签
    :param ax: 子图的画布
    :param type: 数据集类型，用以输出图像标题
    :return: 无
    '''
    zero_type= Tex[np.argwhere(Tey==0)].T
    one_type = Tex[np.argwhere(Tey==1)].T
    if not (ax==None):
        ax.scatter(zero_type[0],zero_type[1],c='blue')
        ax.scatter(one_type[0],one_type[1],c='red')
        ax.set_title(type)
        return ax
    else:
        plt.scatter(zero_type[0],zero_type[1],c='blue')
        plt.scatter(one_type[0],one_type[1],c='red')
        plt.title(type)

nb_diff()
data_diff()
