import matplotlib.pyplot as plt
from functions import *
from grad_des import grad_des
from newton_des import newton_des
import numpy as np
import os

IMAGE_PATH = './Image'
if(not os.path.exists(IMAGE_PATH)):
    os.mkdir(IMAGE_PATH)

def run_des(epoch=100,lr_list=[1,0.1,0.01,0.001,0.0001,0.00001],des_type='grad'):
    if(des_type=='grad'):
        X,y = grad_des(epoch,lr_list)
    elif(des_type=='newton'):
        X,y = newton_des(epoch)

    paint(X[0],X[1],y,des_type=des_type)

def paint(X,Y,Z,des_type):


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X_ = np.arange(min(0,np.min(X)), max(1,np.max(X)), 0.1)
    #print(X.min(0),np.min(X),min(-1,np.min(X[0])),max(1,np.max(X[0])))
    Y_ = np.arange(min(0,np.min(Y)), max(1,np.max(Y)), 0.1)
    X_, Y_ = np.meshgrid(X_, Y_)
    Z_ = f(X_,Y_)
    #print(Z)
    ax.plot_surface(X_, Y_, Z_, color='palegreen',linewidth=0, antialiased=False,alpha=0.3)
    ax.scatter(X,Y,Z,color='navy')
    ax.plot(X,Y,Z,color='slateblue')
    ax.scatter(X[0],Y[0],Z[0],color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    plt.title('Optimization - '+des_type)
    plt.savefig(os.path.join(IMAGE_PATH,des_type+'.jpg'))
    plt.show()

if __name__=='__main__':
    run_des()
    run_des(des_type='newton')