from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np

'''
对MNIST数据集进行分析
'''

#读取数据集
mnist = fetch_openml('mnist_784',data_home='./Datasize',)
X_train,X_test,y_train,y_test=train_test_split(mnist['data'],mnist['target'],train_size=1200,test_size=100,shuffle=True)


#标签计数可视化
def paint_label_counts():
    df = pd.value_counts(y_train)
    index=list(df.index)
    value=list(df.values)
    print(df)
    index = [int(x) for x in index]

    plt.bar(index,value,color='limegreen')
    plt.xlabel('label')
    plt.ylabel('count')
    plt.savefig('Image/mnist_random_count.png')
    plt.show()

#预测错误的数据集可视化
def show_wrong_results():
    svm=SVC(kernel='rbf')
    svm.fit(X_train,y_train)
    y_pred=svm.predict(X_test)
    print(y_pred)
    print(y_test)
    id=0
    for i in range(len(y_pred)):
        if(not int(y_pred[i])==y_test.iloc[i]):
            print(int(y_pred[i]),y_test.iloc[i])
            id=i
            break;


    im=X_test.iloc[id]
    print(type(im))
    im=np.array(im)
    im.resize(28,28)
    plt.imshow(im,cmap='gray')
    plt.axis('off')
    plt.title('label: '+str(y_test.iloc[id])+' pred: '+str(y_pred[id]))
    plt.savefig('Image/show_wrong_results.png')
    plt.show()

show_wrong_results()
