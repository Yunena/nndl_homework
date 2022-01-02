from Models import *
from Mydataset import MyDataset
import pandas as pd
from torch.utils.data import DataLoader
from torchmetrics import Accuracy,R2Score,ConfusionMatrix,AUROC
import matplotlib.pyplot as plt
import os
from torch.optim import lr_scheduler

'''
训练神经网络
'''

class Train:
    def __init__(self,filepath,epoch,train_num,test_num,window,batch_size,gpu_used = False):
        self.filepath = filepath
        self.epoch = epoch
        self.file = pd.read_excel(filepath,header=0,index_col=0)
        self.data_length = len(self.file.columns)
        self.window = window
        self.train_num = train_num
        self.gpu_used = gpu_used
        self.train_data = MyDataset(
            filepath=filepath,
            idlist=range(10-window,train_num),
            window=window
        )
        self.test_data = MyDataset(
            filepath=filepath,
            idlist=range(train_num,train_num+test_num),
            window=window
        )
        self.train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=batch_size,
            shuffle=False
        )
        self.test_loader = DataLoader(
            dataset=self.test_data,
            batch_size=test_num,
            shuffle=False
        )
        self.name_dic={
            1:'Maxtemp',
            2:'Mintemp',
            3:'Rain',
            4:'Wind'
        }

    #训练和测试
    def train(self,aimlabel,net_type,type,LR):

        EPOCH = self.epoch
        net = self.get_net(net_type,type)
        if self.gpu_used:
            net = torch.nn.DataParallel(net, device_ids=[0])
            net = net.cuda()
        if type==1:
            loss_func = nn.MSELoss()
        else:
            loss_func = nn.CrossEntropyLoss()
        if self.gpu_used:
            loss_func = loss_func.cuda()
        opti = torch.optim.Adam(net.parameters(),lr=LR)
        sche = lr_scheduler.ReduceLROnPlateau(opti, mode='min', factor=0.8, patience=5)
        for epoch in range(EPOCH):
            for step,sample in enumerate(self.train_loader):
                b_x = sample['input']
                #print(b_x.shape)
                b_y = sample['output'].T[aimlabel]
                #print(b_y)
                if self.gpu_used:
                    b_x = b_x.cuda()
                    b_y = b_y.cuda()
                out = net(b_x)
                if type==1:
                    loss = loss_func(out,b_y)
                else:
                    loss = loss_func(out,b_y.long())
                opti.zero_grad()
                loss.backward()
                opti.step()
            train_loss = loss.data.cpu().numpy() if self.gpu_used else loss.data
            sche.step(train_loss)
            print(epoch,train_loss)
        #以下测试
        net.eval()
        for step,sample in enumerate(self.test_loader):
            b_x = sample['input']
            b_y = sample['output'].T[aimlabel]
            if self.gpu_used:
                b_x = b_x.cuda()
            out = net(b_x)
        if self.gpu_used:
            out = out.data.cpu()
        acc = Accuracy(num_classes=type)
        r2 = R2Score()
        auc = AUROC(num_classes=type)
        out = out.T
        if type==1:
            res = r2(out[0],b_y)
            out = out.numpy()[0]
        else:
            #print(out.T)
            res = auc(out.T,b_y.int())
            print('Accuracy: ',acc(out.T,b_y.int()))
            out = out.T.numpy()
            out = np.argmax(out,axis=1)
        b_y = b_y.numpy()
        #print(out)
        filename=self.name_dic[aimlabel]
        self.paint(out,b_y,res.numpy(),type,net_type,filename)



    #绘制结果
    def paint(self,preds,target,res,type,net_type,filename=None):
        #print(res)
        length = np.array(range(len(target)))
        if type == 1:
            plt.plot(length,target,c='orange',label='target',linestyle='--')
            plt.plot(length,preds,c='dodgerblue',label='preds')
            plt.legend()
            plt.title(filename+' r2score: '+str(res))
        else:
            cm = ConfusionMatrix(num_classes=type)
            preds=preds.astype('int')
            target=target.astype('int')
            cm_np = cm(torch.tensor(preds),torch.tensor(target)).numpy().astype('int')
            #print(cm_np)
            for i in range(len(cm_np)):
                for j in range(len(cm_np[i])):
                    plt.text(j,i,cm_np[i][j])
            plt.imshow(cm_np, cmap=plt.cm.Blues)
            plt.xticks(range(type))
            plt.yticks(range(type))
            plt.xlabel('Preds')
            plt.ylabel('Target')
            plt.title(filename+' AUC: '+str(res))
        if filename is not None:
            if(not os.path.exists('Result')): os.mkdir('Result')
            if(not os.path.exists('Result/'+net_type)): os.mkdir('Result/'+net_type)
            plt.savefig('Result/'+net_type+'/'+filename)
        plt.show()

    #用于网络选择
    def get_net(self,net_type,type):
        if net_type=='RNN':
            net = RNN(
                input_size=self.window,
                output_size = type
            )
        elif net_type=='LSTM':
            net = LSTM(
                input_size=self.window,
                output_size = type
            )
        elif net_type=='GRU':
            net = LSTM(
                input_size=self.window,
                output_size=type
            )
        return net

'main'
DIVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = DIVICES
tr_10 = Train(
    filepath='Data/data.xlsx',
    epoch=100,
    train_num=2181,
    test_num=334,
    window=10,
    batch_size=25,
    gpu_used=True
)


tr_5 = Train(
    filepath='Data/data.xlsx',
    epoch=100,
    train_num=2186,
    test_num=334,
    window=5,
    batch_size=25,
    gpu_used=True
)

tr_3 = Train(
    filepath='Data/data.xlsx',
    epoch=100,
    train_num=2188,
    test_num=334,
    window=3,
    batch_size=25,
    gpu_used=True
)

type_dic={
    1:1,
    2:1,
    3:3,
    4:2
}

for i in range(3,4):
    print(i)
    '''tr_3.train(
        aimlabel=i,
        net_type='GRU',
        type=type_dic[i],
        LR=0.008
    )'''

    tr_5.train(
        aimlabel=i,
        net_type='GRU',
        type=type_dic[i],
        LR=0.008
    )
    '''tr_10.train(
        aimlabel=i,
        net_type='GRU',
        type=type_dic[i],
        LR=0.008
    )'''
