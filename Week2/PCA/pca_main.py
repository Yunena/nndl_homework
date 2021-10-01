from sklearn.datasets import load_boston
from pca1_by_myself import pca1
from pca2_by_package import pca2
import matplotlib.pyplot as plt
import numpy as np
import os

'''
run_pca:按照题目要求进行滑动窗口、步长选择，可以选择调用pca1（自己完成）或者pca2（调包完成）
paint_stack:绘制并保存带状图（含折线图）
paint_bar：绘制并保存柱状图
'''

IMAGE_PATH = './Image'
if(not os.path.exists(IMAGE_PATH)):
    os.mkdir(IMAGE_PATH)


def run_pca(window=10,stride=10,pca_type = 'pca1'):
    '''

    :param window: 滑动窗口大小，默认为10
    :param stride: 步长大小，默认为10
    :param pca_type: 调用哪一个函数，默认为pca1（自己写的）,只能输入'pca1'或者'pca2'
    :return:无
    '''
    bostons = load_boston()
    data = bostons['data'][:500, -6:]
    #print(bostons.keys())
    #print(bostons['DESCR'])
    pc_list=[]
    if(pca_type=='pca1'):
        for i in range(0,len(data),stride):
            pc_list.append(pca1(data[i:i+window]))
    elif(pca_type=='pca2'):
        for i in range(0,len(data),stride):
            pc_list.append(pca2(data[i:i+window]))

    pc_np = np.array(pc_list).T
    paint_stack(pc_np,pca_type)
    paint_bar(pc_np,pca_type)

    #print(pc_np)
    #print(pc_np.shape)

def paint_stack(data,pca_type):
    '''

    :param data: 需要绘制图像的数据（作为y轴），大小为1维，x轴从0开始（在函数内部生成）
    :param pca_type:调用的函数名，用来给图像命名
    :return:无，但是会存储图像
    '''
    pc1=data[0]
    pc2=data[1]
    pc1_2 = np.add(pc1,pc2)
    #print(len(pc1_2))
    x = np.array([i for i in range(len(pc1))])
    plt.stackplot(x,pc1_2,colors='#CEAEFA')
    plt.plot(x,pc1_2,c='mediumpurple',label='pc1+pc2')
    plt.stackplot(x,pc1,colors='palegreen')
    plt.plot(x,pc1,c='lime',label='pc1')


    plt.legend()

    #plt.yticks(np.arange(0.5,1,0.1))
    plt.ylim(0.5,1.05)

    plt.xlabel('Index')
    plt.ylabel('Value')

    plt.title('PCA Results(Stack) - '+pca_type)

    plt.savefig(os.path.join(IMAGE_PATH,pca_type+'_stack.png'))

    #print(x,data)
    plt.show()

def paint_bar(data,pca_type):
    '''

    :param data: 需要绘制图像的数据（作为y轴），大小为1维，x轴从0开始（在函数内部生成）
    :param pca_type:调用的函数名，用来给图像命名
    :return:无，但是会存储图像
    '''
    pc1=data[0]
    pc2=data[1]
    x = np.array([i for i in range(len(pc1))])
    plt.bar(x,pc1,color='palegreen',label='pc1')
    plt.bar(x,pc2,color='#CEAEFA',label='pc2',bottom=pc1)


    plt.legend()

    #plt.yticks(np.arange(0.5,1,0.1))
    plt.ylim(0.5,1.05)

    plt.xlabel('Index')
    plt.ylabel('Value')

    plt.title('PCA Results(Bar) - '+pca_type)

    plt.savefig(os.path.join(IMAGE_PATH, pca_type + '_bar.png'))

    #print(x,data)
    plt.show()

#main函数
if __name__=='__main__':
    run_pca()
    run_pca(pca_type='pca2')




