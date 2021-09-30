import numpy as np
from functions import *







def grad_des(epoch=100,lr_list=[1,0.1,0.01,0.001,0.0001,0.00001]):
    init_X=np.random.rand(2)
    print('Init:',init_X)
    lr_np = np.array(lr_list)
    lr_len = len(lr_list)
    X_list=[]
    y_list=[]

    X = init_X
    X_list.append(X)
    y_list.append(f(X[0],X[1]))
    print('Current Spot:',X,'Current Height:',f(X[0],X[1]))
    for i in range(epoch):
        grad_X = grad(X[0],X[1])
        #print(grad_X)
        #print(np.array(lr_list)*grad_X[0])
        #print(f(X[0]*np.ones(lr_len)-lr_np*grad_X[0],X[1]*np.ones(lr_len)-lr_np*grad_X[1]))
        lr = lr_list[np.argmin(f(X[0]*np.ones(lr_len)-lr_np*grad_X[0],X[1]*np.ones(lr_len)-lr_np*grad_X[1]))]
        X = X-lr*grad_X
        y = f(X[0],X[1])
        print('Current LR:',lr,'Current Spot:',X,'Current Height:',y)
        if y==float('inf') or y==-float('inf'):
            break
        X_list.append(X)
        y_list.append(y)
    return np.array(X_list).T,np.array(y_list)


#grad_des()


'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)
Z = f(X,Y)
#print(Z)
surf = ax.plot_surface(X, Y, Z, color='forestgreen',linewidth=0, antialiased=False)
plt.show()
'''

'''
    X_np = np.array(X_list)
    y_np = np.array(y_list)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(1, 2, 0.01)
    Y = np.arange(1, 2, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)
    # print(Z)
    ax.plot_surface(X, Y, Z, color='palegreen', linewidth=0, antialiased=False)
    ax.scatter(X_np.T[0],X_np.T[1],y_np,c='b')
    plt.show()
'''