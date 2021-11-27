import torch.nn as nn


'''
存模型
'''

class FC(nn.Module):
    def __init__(self,layer=3,n_input=784,start_n_hidden=500,n_output=10,rate = 0.8):
        super(FC,self).__init__()
        self.input_layer=nn.Linear(n_input,start_n_hidden)
        self.relu = nn.ReLU()
        self.net = nn.Sequential()
        n_hidden = start_n_hidden
        for i in range(layer-1):
            self.net.add_module('fc'+str(i+1),nn.Linear(n_hidden,int(n_hidden*rate)))
            self.net.add_module('relu'+str(i+1),nn.ReLU())
            n_hidden = int(n_hidden * rate)

        self.out_layer = nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.net(x)
        out = self.out_layer(x)
        return out

class CNN_3layers(nn.Module):
    def __init__(self, input_size=None, out_channel_list=[int(16 * 2 ** i) for i in range(3)],kernel_size=7):
        super(CNN_3layers,self).__init__()
        if input_size is None:
            input_size = [28, 28]
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channel_list[0],
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size-1)/2)
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(out_channel_list[0],out_channel_list[1],kernel_size,1,int((kernel_size-1)/2)),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(out_channel_list[1],out_channel_list[2],kernel_size,1,int((kernel_size-1)/2)),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc=nn.Linear(out_channel_list[2]*int(input_size[0]/8)*int(input_size[1]/8),10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

class CNN_11layers(nn.Module):
    def __init__(self, input_size=None, out_channel_list=[int(16 * 2 ** int(i / 2)) for i in range(11)],kernel_size=7):
        super(CNN_11layers,self).__init__()
        if input_size is None:
            input_size = [28, 28]
        self.conv1=nn.Sequential(nn.Conv2d(1,out_channel_list[0],kernel_size,1,int((kernel_size-1)/2)),nn.ReLU())
        self.cnnnet = nn.Sequential()
        for i in range(10):
            if(i%3==0):
                self.cnnnet.add_module('conv'+str(i+2),nn.Sequential(nn.Conv2d(out_channel_list[i],out_channel_list[i+1],kernel_size,1,int((kernel_size-1)/2)),nn.ReLU(),nn.MaxPool2d(2)))
            else:
                self.cnnnet.add_module('conv'+str(i+2),nn.Sequential(nn.Conv2d(out_channel_list[i],out_channel_list[i+1],kernel_size,1,int((kernel_size-1)/2)),nn.ReLU()))

        self.fc = nn.Linear(out_channel_list[10]*int(input_size[0]/16)*int(input_size[1]/16),10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.cnnnet(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out





