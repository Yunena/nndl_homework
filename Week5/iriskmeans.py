import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score


'''
cluster_ploting(target,cluster,name):降维绘图，便于可视化
kmeans_cluster(clusters):聚类
test_cluster(N):测试聚类个数的影响
'''

#导入数据和模型
iris = load_iris()
data=iris.data
targ=iris.target
pca = PCA(n_components=2)

#print(iris)


def cluster_ploting(target=targ,clusters=3,name='Original Target',title=''):
    '''

    :param target:预测的或真实的标签（默认为真实标签）
    :param clusters:预测的类别数（默认为真实的3类）
    :param name:图片命名（默认为表示真实标签）
    :param title:图片标题，用于存轮廓系数（默认为空）
    :return:
    '''
    data_plot=pca.fit_transform(data)
    #print(data_plot.shape)
    for i in range(clusters):
        #print(i)
        data_plot_id=data_plot[np.argwhere(target==i)]
        #Tex[np.argwhere(Tey == 0)].T
        plt.scatter(data_plot_id.T[0],data_plot_id.T[1])
    plt.title(str(clusters)+' clusters '+title)
    plt.savefig('Image/'+name)
    plt.show()



def kmeans_cluster(clusters):
    '''

    :param clusters: 聚类的个数
    :return: 轮廓系数
    '''
    kmeans=KMeans(n_clusters=clusters)
    kmeans.fit(data)
    pred=kmeans.labels_
    silhscore=silhouette_score(data,pred)
    cluster_ploting(pred,clusters,name=str(clusters),title=str(silhscore))
    print(silhscore)
    return silhscore

#kmeans_cluster(6)
def test_cluster(N):
    '''

    :param N: 从2开始，需要聚类的个数
    :return:
    '''
    scores=[]
    for i in range(2,N+1):
        scores.append(kmeans_cluster(i))
    plt.plot(range(2,N+1),scores,c='forestgreen')
    plt.xticks(range(2,N+1))
    plt.xlabel('cluster number')
    plt.ylabel('silhouette scores')
    plt.title('Silhouette Scores')
    plt.savefig('Image/cluster test')
    plt.show()


test_cluster(7)



