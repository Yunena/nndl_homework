from sklearn.naive_bayes import *
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

gaus = GaussianNB()
bern = BernoulliNB()
muti = MultinomialNB()
comp = ComplementNB()

moons = make_moons(500)
circles = make_circles(500)
classfication = make_classification(500)






def nb_diff():
    Trx,Tex,Try,Tey = train_test_split(moons[0],moons[1],train_size=0.7)
    nb_paint(Tex,Tey,type='Moon Target')
    #plt.title('Moon Target')
    plt.savefig('./Image/moon_test_target.png')
    plt.show()

    gaus.fit(Trx,Try)
    bern.fit(Trx,Try)

    gaus_y = gaus.predict(Tex)
    bern_y = bern.predict(Tex)
    fig = plt.figure()

    plt.title('Predicted Results')

    ax1= fig.add_subplot(121)
    nb_paint(Tex,gaus_y,ax1,'Gaussian')
    ax2= fig.add_subplot(122)
    nb_paint(Tex,bern_y,ax2,'Bernoulli')
    plt.savefig('./Image/moon_results.png')
    plt.show()

def data_diff():
    Trx, Tex, Try, Tey = train_test_split(moons[0], moons[1], train_size=0.7)
    gaus = GaussianNB()
    gaus.fit(Trx,Try)
    y=gaus.predict(Tex)
    paint_data_diff(Tex,Tey,y,'Moons')
    print(accuracy_score(Tey,y))

    Trx, Tex, Try, Tey = train_test_split(circles[0], circles[1], train_size=0.7)
    gaus = GaussianNB()
    gaus.fit(Trx, Try)
    y = gaus.predict(Tex)
    paint_data_diff(Tex, Tey, y, 'Circles')
    print(accuracy_score(Tey,y))

    Trx, Tex, Try, Tey = train_test_split(classfication[0], classfication[1], train_size=0.7)
    gaus = GaussianNB()
    gaus.fit(Trx, Try)
    y = gaus.predict(Tex)
    paint_data_diff(Tex, Tey, y, 'Classfication')
    print(accuracy_score(Tey,y))

def paint_data_diff(Tex,Tey,y,type):
    fig = plt.figure()

    plt.title(type+' results')

    ax1= fig.add_subplot(121)
    nb_paint(Tex,Tey,ax1,'Targets')
    ax2= fig.add_subplot(122)
    nb_paint(Tex,y,ax2,'Results')
    plt.savefig('./Image/'+type+'.png')
    plt.show()

def nb_paint(Tex,Tey,ax=None,type='gaus'):
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


data_diff()
