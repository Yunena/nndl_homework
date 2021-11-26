from models import *
import torch
import torch.nn as nn
import os
import torchvision
import torch.utils.data as Data
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
import numpy as np

'''
参数比较
'''

GPU_USED=True

#创建图像文件夹
if not(os.path.exists('./Image/')):
    os.mkdir('./Image/')


#导入数据
DOWNLOAD = False
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=150,
    shuffle=True
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD
)

test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=2000,
    shuffle=False
)

def training(net,LR):
    EPOCH=10
    loss_func = nn.CrossEntropyLoss()
    opti = torch.optim.Adam(net.parameters(),lr=LR)
    for epoch in range(EPOCH):
        print(epoch,end=' ')
        for step,(b_x,b_y) in enumerate(train_loader):
            print(step, end=' ')
            out = net(b_x)
            loss = loss_func(out,b_y)
            opti.zero_grad()
            loss.backward()
            opti.step()
        print()

    net.eval()
    t_x=None
    target = None
    for (p_x,p_y) in test_loader:
        t_x = p_x
        target = p_y
    preds = net(t_x)
    accuracy = Accuracy(num_classes=10)
    acc = accuracy(preds,target)
    return acc.numpy(),sum(p.numel() for p in net.parameters())

def gputraining(net,LR):
    DIVICES = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = DIVICES
    net = torch.nn.DataParallel(net, device_ids=[i for i in range(len(DIVICES.split(',')))])
    net = net.cuda()

    EPOCH = 10
    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.cuda()
    opti = torch.optim.Adam(net.parameters(),lr=LR)
    for epoch in range(EPOCH):
        print(epoch,end=' ')
        for step,(b_x,b_y) in enumerate(train_loader):
            print(step, end=' ')
            out = net(b_x.cuda())
            loss = loss_func(out,b_y.cuda())
            opti.zero_grad()
            loss.backward()
            opti.step()
        print()

    net.eval()
    t_x=None
    target = None
    for (p_x,p_y) in test_loader:
        t_x = p_x.cuda()
        target = p_y.cuda()
    preds = net(t_x)
    accuracy = Accuracy(num_classes=10)
    acc = accuracy(preds,target)
    return acc.data.cpu().numpy(),sum(p.numel() for p in net.parameters())

def fc_training():
    start_n_hidden_list=[1024,512,256,1000,500]
    rate_list=[1,1,1,0.8,0.8]
    acclist=[]
    numlist=[]
    for i in range(len(rate_list)):
        net = FC(start_n_hidden=start_n_hidden_list[i],rate=rate_list[i])
        acc,num=gputraining(net,0.0001) if(GPU_USED) else training(net,0.0001)
        print(num,acc)
        acclist.append(acc)
        numlist.append(num)
        del(net)
    accnp = np.array(acclist)
    numnp = np.array(numlist)
    idx = np.argsort(numnp)
    return numnp[idx],accnp[idx]

def cnn3_training():
    out_channel_list_list=[
        [int(16 * 2 ** i) for i in range(3)],
        [int(32 * 2 ** i) for i in range(3)],
        [int(16 * 2 ** i) for i in range(3)],
        [int(32 * 2 ** i) for i in range(3)],
        [int(16 * 2 ** i) for i in range(3)]
    ]
    kernel_size_list=[7,7,5,5,9]
    acclist=[]
    numlist=[]
    for i in range(len(kernel_size_list)):
        net = CNN_3layers(out_channel_list=out_channel_list_list[i],kernel_size=kernel_size_list[i])
        acc,num=gputraining(net,0.0001) if(GPU_USED) else training(net,0.0001)
        print(num,acc)
        acclist.append(acc)
        numlist.append(num)
        del(net)

    accnp = np.array(acclist)
    numnp = np.array(numlist)
    idx = np.argsort(numnp)
    return numnp[idx],accnp[idx]

def cnn11_training():
    out_channel_list_list=[
        [int(16 * 2 ** int(i/2)) for i in range(11)],
        [int(32 * 2 ** int(i/2)) for i in range(11)],
        [int(16 * 2 ** int(i/2)) for i in range(11)],
        [int(32 * 2 ** int(i/2)) for i in range(11)],
        [int(16 * 2 ** int(i/2)) for i in range(11)]
    ]
    kernel_size_list=[7,7,5,5,9]
    acclist=[]
    numlist=[]
    for i in range(len(kernel_size_list)):
        net = CNN_11layers(out_channel_list=out_channel_list_list[i],kernel_size=kernel_size_list[i])
        acc,num=gputraining(net,0.0001) if(GPU_USED) else training(net,0.0001)
        print(num,acc)
        acclist.append(acc)
        numlist.append(num)
        del(net)

    accnp = np.array(acclist)
    numnp = np.array(numlist)
    idx = np.argsort(numnp)
    return numnp[idx],accnp[idx]

def alltraining():
    numlist,acclist=fc_training()
    plt.scatter(numlist,acclist,c='r',marker='o')
    plt.plot(numlist,acclist,c='r',label='FC')
    numlist,acclist=cnn3_training()
    plt.scatter(numlist,acclist,c='b',marker='s')
    plt.plot(numlist,acclist,c='b',label='CNN3layers')
    numlist,acclist=cnn11_training()
    plt.scatter(numlist,acclist,c='g',marker='^')
    plt.plot(numlist,acclist,c='g',label='CNN11layers')

    plt.xlabel('parameters')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig('Image/parametertest')
    plt.show()

alltraining()
#cnn3_training()

