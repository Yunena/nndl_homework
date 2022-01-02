import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
'''
读取数据集的方式
'''


class MyDataset(Dataset):
    def __init__(self,filepath,idlist,window):
        self.datalist = pd.read_excel(filepath,header=0,index_col=0)
        self.idlist = idlist
        self.window = window



    def __len__(self):
        return len(self.idlist)

    def __getitem__(self,idx):
        id = self.idlist[idx]
        input_list = []
        for i in range(self.window):
            #input_list.append(torch.Tensor(list(self.datalist.iloc[id+i,:])))
            input_list.append(list(self.datalist.iloc[id + i, :]))
        input_np = np.array(input_list)
        input_np = input_np.T

        output = self.datalist.iloc[id+self.window,:]
        sample = {
            "input":torch.Tensor(input_np),
            #"input": input_list,
            "output":torch.Tensor(output)
        }

        return sample


