from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #必须标准化

'''
用SGD对波士顿房价进行预测
training():训练并预测结果
evaluation():进行评估
ploting():绘图

'''


#准备并分配数据
bostons=load_boston()
target = bostons['target']
data = bostons['data']
#print(bostons['feature_names'])
Xtrain,Xtest,ytrain,ytest=train_test_split(data,target,train_size=0.7)
#标准化数据
ssx=StandardScaler()
ssy=StandardScaler()
Xtrain=ssx.fit_transform(Xtrain)
Xtest=ssx.transform(Xtest)
ytrain=ssy.fit_transform(ytrain.reshape(-1,1))
#ytest=ssy.transform(ytest.reshape(-1,1))

def training():
    '''

    :return: ypred:（标准化的）预测结果
             ssy.transform(ytest.reshape(-1,1)):（标准化的）真实标签
    '''
    sgd=SGDRegressor(max_iter=1000)
    sgd.fit(Xtrain,ytrain)
    ypred=sgd.predict(Xtest).astype('float')
    return ypred,ssy.transform(ytest.reshape(-1,1))

def evaluation(ypred,ytest):
    '''

    :param ypred:预测值
    :param ytest:真实值
    :return:SSE值
    '''
    return r2_score(ytest,ypred)

def ploting(ypred,ytest,r2=0,type='std'):
    '''

    :param ypred: 预测值
    :param ytest: 真实值
    :param r2: SSE值
    :param type: 用于文件命名
    :return:
    '''
    #acc = 0
    #acc=evaluation(ypred,ytest)
    #print(ytest.shape,ypred.shape)
    plt.scatter(ypred,ytest,10,'b')
    plt.plot(ytest,ytest,'r')
    plt.xlabel('predicted values')
    plt.ylabel('targets')
    plt.title('Test r2: '+str(r2))
    plt.savefig('Image/'+type+'.png')
    plt.show()

if __name__=='__main__':
    p,t=training()
    std_r2=evaluation(p,t)
    r2=evaluation(ssy.inverse_transform(p),ytest)
    print('MSE: ',mean_squared_error(t,p))
    ploting(p,t,std_r2,'std_r2')
    ploting(ssy.inverse_transform(p),ytest,r2,'r2')

