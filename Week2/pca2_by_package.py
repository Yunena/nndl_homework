from sklearn.decomposition import PCA
'''
pca2：调用sklearn的PCA完成
'''


def pca2(data,n_component=2):
    '''
    :param data: array, 大小为 (n_examples,n_features),n_examples是用例个数， n_features 是特征个数
    :param n_component: int,降维维数，默认输入为2
    :return:array, 返回前n个排好序的信息占比，n为输入的n_component
    '''
    pca = PCA(n_components=n_component)
    pca.fit(data)
    eig_val_sort = pca.explained_variance_ratio_
    return eig_val_sort


