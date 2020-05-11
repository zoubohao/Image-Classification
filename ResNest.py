import torch
import torch.nn as nn
import torch.nn.functional as F
from Tools import AddN
from Tools import Conv2dDynamicSamePadding

class Split_Attention(nn.Module):

    def __init__(self,r,in_channels,inner_channels):
        super().__init__()
        self.r = r
        self.in_channels = in_channels
        self.dense1 = nn.Sequential(nn.Linear(in_channels,inner_channels,bias=True),
                                    nn.BatchNorm1d(inner_channels,eps=0.001,momentum=0.01),
                                    nn.ReLU(inplace=True))
        self.dense2 = nn.Linear(inner_channels,in_channels * r,bias=True)


    def forward(self,inputs):
        addedTensor = AddN(inputs)
        globalPooling = F.adaptive_avg_pool2d(addedTensor,output_size=[1,1]).view(-1,self.in_channels)
        dense1 = self.dense1(globalPooling)
        dense2List = torch.chunk(self.dense2(dense1),chunks=self.r,dim=-1)
        attentionList = []
        for i,oneDense in enumerate(dense2List):
            softMaxT = torch.softmax(oneDense,dim=-1).unsqueeze(-1).unsqueeze(-1)
            attentionList.append(softMaxT * inputs[i])
        return AddN(attentionList)

from collections import OrderedDict
class Cardinal_Block(nn.Module):

    def __init__(self,r,in_channels):
        super().__init__()
        self.r = r
        ### Conv1
        self.conv1 = Conv2dDynamicSamePadding(in_channels,in_channels,1,1,groups=r,bias=False)
        bn1Ordic = OrderedDict()
        for i in range(r):
            bn1Ordic["BN_A1" + str(i)] = nn.Sequential(nn.BatchNorm2d(in_channels // r, eps=0.001, momentum=0.01),nn.ReLU(inplace=True))
        self.bn1Dic = nn.ModuleDict(bn1Ordic)
        ### Conv3
        self.conv3 = Conv2dDynamicSamePadding(in_channels,in_channels * r,3,1,groups=r,bias=False)
        bn2Oridic = OrderedDict()
        for i in range(r):
            bn2Oridic["BN_A2" + str(i)] = nn.Sequential(nn.BatchNorm2d(in_channels,eps=0.001,momentum=0.01),nn.ReLU(inplace=True))
        self.bn2Dic = nn.ModuleDict(bn2Oridic)
        ### Split
        self.split_attention = Split_Attention(r,in_channels,inner_channels=in_channels * 2)

    def forward(self,x):
        conv1 = self.conv1(x)
        splitT1 = torch.chunk(conv1,chunks=self.r,dim=-3)
        bn1T = []
        for i in range(self.r):
            bn1T.append(self.bn1Dic["BN_A1" + str(i)](splitT1[i]))
        conv3 = self.conv3(torch.cat(bn1T,dim=-3))
        splitT2 = torch.chunk(conv3,chunks=self.r,dim=-3)
        bn2T = []
        for i  in range(self.r):
            bn2T.append(self.bn2Dic["BN_A2" + str(i)](splitT2[i]))
        return  self.split_attention(bn2T)

class ResNest_Block(nn.Module):

    def __init__(self,k = 4,r= 4,in_channels=16,drop_p=0.2):
        super().__init__()
        self.k = k
        self.cardinalList = nn.ModuleList([Cardinal_Block(r, in_channels // k) for _ in range(k)])
        self.conv1 = nn.Sequential(Conv2dDynamicSamePadding(in_channels,in_channels,1,1,bias=False),
                                   nn.BatchNorm2d(in_channels,eps=0.001,momentum=0.01))
        self.dropout = nn.Dropout2d(drop_p,inplace=True)

    def forward(self,x):
        oneCardinalT = torch.chunk(x,chunks=self.k,dim=-3)
        catList = []
        for i,cardinalM in enumerate(self.cardinalList):
            catList.append(cardinalM(oneCardinalT[i]))
        catTensor = torch.cat(catList,dim=-3)
        return self.dropout(x.clone() + self.conv1(catTensor))

if __name__ == "__main__":
    testInput = torch.randn(size=[5,16,31,31]).float()
    testModule = ResNest_Block(k = 4,r= 4,in_channels=16,drop_p=0.2)
    testRes = testModule(testInput)
    testRes.mean().backward()
    from Tools import plot_grad_flow
    plot_grad_flow(testModule.named_parameters())
    print(testModule(testInput).shape)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    writer.add_graph(testModule, testInput)
    writer.close()
    testTensor = torch.rand(size=[3,6]).float()










