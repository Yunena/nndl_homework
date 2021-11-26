from models import FC
import torchvision
import os
import torch.utils.data as Data
import torch.nn as nn
import torch
from torchmetrics import Accuracy
import matplotlib.pyplot as plt

'''
全连接的深度的影响
'''
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




def training(layers=3):
    EPOCH = 10
    net = FC(layer=layers,rate=1)
    loss_func = nn.CrossEntropyLoss()
    LR=0.0001
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
    t_x = test_data.test_data.type(torch.FloatTensor)
    target = test_data.test_labels
    preds = net(t_x)
    accuracy = Accuracy(num_classes=10)
    acc = accuracy(preds,target)
    return acc

def deepevaluate():
    deeplist = range(1,16)
    acclist= []
    for deep in deeplist:
        acc = training(layers=deep)
        print('layers:',deep,'|accuracy:',acc)
        acclist.append(acc)

    plt.plot(deeplist,acclist,c='limegreen')
    plt.xlabel('layers')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.savefig('Image/deeptest')
    plt.show()

deepevaluate()





