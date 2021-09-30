import numpy as np

def f(x,y):
    return x**3+y**3-3*x*y

def f_x(x,y):
    return 3*(x**2)-3*y

def f_y(x,y):
    return 3*(y**2)-3*x

def f_xx(x,y):
    return 6*x

def f_xy(x,y):
    return 3

def f_yy(x,y):
    return 6*y

def grad(x,y):
    return np.array([f_x(x,y),f_y(x,y)])

def hessian(x,y):
    np_r = np.zeros((2,2))
    #print(np_r)
    np_r[0][0]=f_xx(x,y)
    np_r[0][1]=f_xy(x,y)
    np_r[1][0]=f_xy(x,y)
    np_r[1][1]=f_yy(x,y)
    return np_r