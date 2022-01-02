from hmmlearn import hmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

'''
应用HMM模型
'''
class HMMtrain:
    def __init__(self,filename,tempidx=1):
        self.df = pd.read_excel(filename,header=0,index_col=0)
        self.maxtemp, self.mintemp = self.get_sequence()
        self.A = self.get_A(tempidx)
        self.B = self.get_B(tempidx)
        self.pi = self.get_pi(tempidx)

    #获取HMM模型需要输出的参数
    def get_sequence(self):
        maxtemp = np.array(list(self.df.iloc[:,1]))
        mintemp = np.array(list(self.df.iloc[:,2]))

        diff_max_temp = np.zeros(len(maxtemp))
        diff_min_temp = np.zeros(len(mintemp))
        for i in range(1,len(maxtemp)):
            diff = maxtemp[i]-maxtemp[i-1]
            if diff<0:
                diff_max_temp[i]=-1
            elif diff==0:
                diff_max_temp[i]=0
            else:
                diff_max_temp[i]=1

        for i in range(1,len(mintemp)):
            diff = mintemp[i]-mintemp[i-1]
            if diff<0:
                diff_min_temp[i]=-1
            elif diff==0:
                diff_min_temp[i]=0
            else:
                diff_min_temp[i]=1

        return diff_max_temp,diff_min_temp


    #获取HMM需要的参数（实际使用的时候并没有用）
    def get_A(self,tempidx,cate = 3,idx = 3):
        A = np.zeros((cate,cate),dtype='float64')
        cate_list = [-1,0,1]
        ori_list = self.maxtemp if tempidx==1 else self.mintemp
        for i in cate_list:
            cntt = np.sum(ori_list[:-1]==i)
            for j in cate_list:
                cnt = 0
                for k in range(1,len(ori_list)):
                    if ori_list[k]==j and ori_list[k-1]==i:
                        cnt += 1
                A[i][j]=float(cnt)/float(cntt)
        print(A)
        return A

    #获取HMM需要的参数（实际使用的时候并没有用）
    def get_B(self,tempidx,idx = 3,cate=3):
        aimarray = self.maxtemp if tempidx==1 else self.mintemp
        aim_list = [-1,0,1]
        cate_list = range(cate)
        ori_list = list(self.df.iloc[:,idx])
        length = len(ori_list)
        B = np.zeros((cate,3))
        for i in aim_list:
            cntt = np.sum(aimarray==i)
            for j in cate_list:
                cnt = 0
                for k in range(length):
                    if ori_list[k]==j and aimarray[k]==i:
                        cnt += 1
                B[i][j]=float(cnt)/float(cntt)
        print(B)
        return B

    #获取HMM需要的参数（实际使用的时候并没有用）
    def get_pi(self,tempidx,cate=3,idx=3):
        ori_list = self.maxtemp if tempidx==1 else self.mintemp
        pi = np.zeros(cate,dtype='float64')
        cate_list = [-1,0,1]
        length = len(ori_list)
        for i in cate_list:
            pi[i] = float(np.sum(np.array(ori_list)==i))/length
        print(pi)
        return pi

    #训练
    def train(self):
        model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
        X_train = np.array(list(self.df.iloc[:,3])[:2191]).reshape(-1,1)
        X_test = np.array(list(self.df.iloc[:,3])[2191:]).reshape(-1,1)
        print(X_test.reshape(1,-1))
        model.fit(X_train)
        z=model.predict(X_train)
        print(z)


        print(accuracy_score(z-1,self.maxtemp[:2191]))
        self.paint(z-1,self.maxtemp[:2191],None)


    #绘制结果
    def paint(self,preds,target,name):
        length = range(len(preds))
        plt.plot(length,target,c='orange',label='target',linestyle='--')
        plt.plot(length,preds,c='dodgerblue',label='preds')
        plt.yticks([-1,0,1])
        plt.legend()
        plt.title(name)
        if name is not None:
            plt.savefig('Result/'+name)

        plt.show()




#main
ht = HMMtrain(filename='Data/data.xlsx')
ht.train()