import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
'''
神经网络模型：RNN，LSTM，GRU
'''
class RNN(nn.Module):
    def __init__(self,input_size,output_size):
        super(RNN,self).__init__()
        self.hidden = nn.RNN(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.FC = nn.Linear(64*5,output_size)

    def forward(self,x):
        out,_ = self.hidden(x,None)
        out = out.contiguous().view(out.size(0), -1)
        out = F.sigmoid(out)
        out = self.FC(out)
        return out

class LSTM(nn.Module):
    def __init__(self,input_size,output_size):
        super(LSTM,self).__init__()
        self.hidden = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.FC = nn.Linear(64*5,output_size)

    def forward(self,x):
        out,_ = self.hidden(x,None)
        out = out.contiguous().view(out.size(0), -1)
        out = F.sigmoid(out)
        out = self.FC(out)
        return out

class GRU(nn.Module):
    def __init__(self,input_size,output_size):
        super(GRU,self).__init__()
        self.hidden = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.FC = nn.Linear(64*5,output_size)

    def forward(self,x):
        out,_ = self.hidden(x,None)
        out = out.contiguous().view(out.size(0), -1)
        out = F.sigmoid(out)
        out = self.FC(out)
        return out