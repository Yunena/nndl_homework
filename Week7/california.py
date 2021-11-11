from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

cal=fetch_california_housing()
data=cal.data
target=cal.target
X_train,X_test,y_train,y_test=train_test_split(data,target,train_size=0.7)
ssx=StandardScaler()
ssy=StandardScaler()
stdX_train=ssx.fit_transform(X_train)
stdX_test=ssx.transform(X_test)
stdy_train=y_train
stdy_test=y_test


#探索数据集
def explore_dataset():
    print(cal.feature_names)
    print('max', np.max(data, axis=0))
    print('min', np.min(data, axis=0))
    print('median',np.median(data,axis=0))
    print('iqr',np.quantile(data,q=0.75,axis=0,interpolation='higher')-np.quantile(data,q=0.25,axis=0,interpolation='lower'))
    print('mean',np.mean(data,axis=0))
    print('max', np.max(target, axis=0))
    print('min', np.min(target, axis=0))
    print('median',np.median(target,axis=0))
    print('iqr',np.quantile(target,q=0.75,axis=0,interpolation='higher')-np.quantile(target,q=0.25,axis=0,interpolation='lower'))
    print('mean',np.mean(target,axis=0))


#explore_dataset()

#adjuser2
def adjusted_r2(y_true,y_pred,n,p):
    '''
    :param y_true: 真实值
    :param y_pred: 预测值
    :param n: 样本量
    :param p: 样本特征数
    :return:adjust_r2
    '''
    #print(n,p)
    r2=r2_score(y_true,y_pred)
    ar2=1-(1-r2)*(n-1)/(n-p-1)
    return ar2

#多元线性回归
def linear_reg():
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    y_pred=lr.predict(X_test)
    mse=mean_squared_error(y_test,y_pred)
    ar2=adjusted_r2(y_test,y_pred,len(y_test),len(cal.feature_names))
    print('mse:',mse,end=' ')
    #print(r2_score(y_test,lr.predict(X_test)))
    print('adjust_r2:',ar2)

#岭回归
def ridge_reg(alpha=0.01,xtr=X_train,xte=X_test,ytr=y_train,yte=y_test):
    ridge=Ridge(alpha=alpha)
    ridge.fit(xtr,ytr)
    y_pred=ridge.predict(xte)
    mse=mean_squared_error(yte,y_pred)
    ar2=adjusted_r2(yte,y_pred,len(yte),len(cal.feature_names))
    print('mse:',mse,end=' ')
    print('adjust_r2:',ar2)
    return ridge.coef_

#Lasso回归
def lasso_reg(alpha=0.01,xtr=X_train,xte=X_test,ytr=y_train,yte=y_test):
    lasso=Lasso(alpha=alpha)
    lasso.fit(xtr,ytr)
    y_pred=lasso.predict(xte)
    mse=mean_squared_error(yte,y_pred)
    ar2=adjusted_r2(yte,y_pred,len(yte),len(cal.feature_names))
    print('mse:',mse,end=' ')
    print('adjust_r2:',ar2)
    return lasso.coef_


#linear_reg()
#分析多重共线性
def test_alpha():
    exp_list=[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
    alpha_list=[10**i for i in exp_list]
    ridge_coef=[]
    lasso_coef=[]
    for alpha in alpha_list:
        print(alpha)
        coef=ridge_reg(alpha,stdX_train,stdX_test,stdy_train,stdy_test)
        #print(coef)
        ridge_coef.append(coef)
        coef=lasso_reg(alpha,stdX_train,stdX_test,stdy_train,stdy_test)
        lasso_coef.append(coef)
    #plt.plot(alpha_list,np.array(lasso_coef))
    if(not os.path.exists('Image/')):os.mkdir('Image/')


    plt.plot([i for i in range(len(exp_list))],ridge_coef)
    plt.xticks([i for i in range(len(exp_list))],[str(exp) for exp in exp_list])
    plt.xlabel('Power of 10')
    plt.ylabel('Coef')
    plt.title('Ridge results')
    plt.savefig('Image/Ridge')
    plt.show()

    plt.plot([i for i in range(len(exp_list))],lasso_coef)
    plt.xticks([i for i in range(len(exp_list))],[str(exp) for exp in exp_list])
    plt.xlabel('Power of 10')
    plt.ylabel('Coef')
    plt.title('Lasso results')
    plt.savefig('Image/Lasso')
    plt.show()

#VIF
def vif_check():
    '''print([1 for i in range(len(data))])
    data['b']=[1 for i in range(len(data))]'''
    data_=np.c_[data,np.ones(len(data))]
    vif = [variance_inflation_factor(data_,i) for i in range(data.shape[1])]
    print(vif)

#vif_check()

if __name__=="__main__":
    print('LinearRegression:')
    linear_reg()
    print('Ridge:')
    ridge_reg()
    print('Lasso:')
    lasso_reg()