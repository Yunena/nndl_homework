import time
from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random

'''
分析MNIST数据
'''

if not os.path.exists('Image/'): os.mkdir('Image/')

start = time.time()
mnist = fetch_openml('mnist_784',data_home='./Datasize')
print('mnist have loaded.',time.time() - start)#显示load时间
data=mnist['data']
target=mnist['target']


#对随机50个数据可视化
def visualize():
    id_list = random.sample(range(len(data)),50)
    for i in range(50):
        id = id_list[i]
        im = data.iloc[id]
        label = target.iloc[id]
        im = np.array(im)
        im.resize(28,28)
        ax = plt.subplot(5,10,i+1)
        ax.imshow(im,cmap='gray')
        ax.axis('off')
        ax.set_title(str(label))
    plt.suptitle('Random 50 Visualization')
    plt.savefig('Image/mnist_visulization')
    plt.show()

#标签数值可视化
def paint():
    df = pd.value_counts(target)
    index=list(df.index)
    value=list(df.values)
    print(df)
    index = [int(x) for x in index]

    plt.bar(index,value,color='limegreen')

    for x,y in zip(index,value):
        plt.text(x + 0.05, y + 0.05, str(y), ha='center', va='bottom')

    plt.xlabel('label')
    plt.xticks(range(10))
    plt.ylabel('count')
    plt.title('Count')
    plt.savefig('Image/mnist_count.png')
    plt.show()

if __name__=="__main__":
    visualize()
    paint()





