import numpy as np

def pca1(data,n_component=2):
    #data should be 2D
    data_T = data.T #transpose
    dim = len(data_T)
    num = len(data)
    norm_data_T = data_T.copy()
    meanvalue_np=np.mean(data_T,axis=1)
    for i in range(dim):
        norm_data_T[i]-=meanvalue_np[i]*np.ones(num)
    cov = np.dot(norm_data_T,norm_data_T.T)/num


    eig_val = np.linalg.eig(cov)[0]
    eig_val_sort = np.sort(eig_val)[::-1]
    eig_val_sort = eig_val_sort/np.sum(eig_val_sort)

    return eig_val_sort[0:n_component]







