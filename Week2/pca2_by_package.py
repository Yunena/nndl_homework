from sklearn.decomposition import PCA


def pca2(data,n_component=2):
    pca = PCA(n_components=n_component)
    pca.fit(data)
    eig_val_sort = pca.explained_variance_ratio_
    return eig_val_sort


