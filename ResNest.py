import torch
import torch.nn as nn
import torch.nn.functional as F
from Tools import AddN
from Tools import Conv2dDynamicSamePadding
import math
from Tools import Pool2dStaticSamePadding
from Tools import Mish
from Tools import drop_connect_B

class Split_Attention(nn.Module):

    def __init__(self,r,in_channels,inner_channels):
        super().__init__()
        self.r = r
        self.in_channels = in_channels
        self.dense1 = nn.Sequential(Conv2dDynamicSamePadding(in_channels,inner_channels,1,1),
                                    nn.BatchNorm2d(inner_channels,eps=1e-3,momentum=1e-2),
                                    Mish())
        self.dense2 = Conv2dDynamicSamePadding(inner_channels,in_channels * r,1,1,groups=r)


    def forward(self,inputs):
        addedTensor = AddN(inputs)
        globalPooling = F.adaptive_avg_pool2d(addedTensor,output_size=[1,1])
        dense1 = self.dense1(globalPooling)
        dense2List = torch.chunk(self.dense2(dense1),chunks=self.r,dim=-3)
        attentionList = []
        for i,oneDense in enumerate(dense2List):
            softMaxT = torch.softmax(oneDense,dim=1)
            attentionList.append(softMaxT * inputs[i])
        return AddN(attentionList)

class Cardinal_Block(nn.Module):

    def __init__(self,r,in_channels):
        super().__init__()
        self.r = r
        ### Conv
        self.conv1 = nn.Sequential(Conv2dDynamicSamePadding(in_channels,in_channels,1,1,groups=r,bias=False),
                                   nn.GroupNorm(num_groups=r,num_channels=in_channels,eps=1e-3,affine=True),
                                   Mish())
        self.conv3 = nn.Sequential(Conv2dDynamicSamePadding(in_channels,in_channels * r,3,1,groups=r,bias=False),
                                   nn.GroupNorm(num_groups=r,num_channels=in_channels * r,eps=1e-3,affine=True),
                                   Mish())
        ### Split
        self.split_attention = Split_Attention(r,in_channels,inner_channels=in_channels * 2)

    def forward(self,x):
        conv1 = self.conv1(x)
        conv3 = self.conv3(conv1)
        splitT = torch.chunk(conv3,self.r,dim=1)
        return  self.split_attention(splitT)

class OneBlock(nn.Module):

    def __init__(self,in_channels,out_channels,k = 2,r= 4,pooling = "AVG",drop_connect_rate = 0.5):
        super().__init__()
        self.k = k
        self.cardinalList = nn.ModuleList([Cardinal_Block(r, in_channels // k) for _ in range(k)])
        self.conv1 = nn.Sequential(Conv2dDynamicSamePadding(in_channels,in_channels,1,1,bias=False),
                                   nn.BatchNorm2d(in_channels,eps=1e-3,momentum=1e-2))
        self.downSample = False
        self.drop_conn = drop_connect_rate
        if in_channels != out_channels:
            ### MUST REMEMBER THAT PLEASE DO DOWN SAMPLING WITH THIS KERNEL SIZE 3 AND STRIDE 2 !!!!!
            ### otherwise, THE NET WORK WOULD NOT CONVERGENCE.
            self.DownSample = nn.Sequential(Pool2dStaticSamePadding(3,2,pooling=pooling),
                Conv2dDynamicSamePadding(in_channels,out_channels,kernel_size=1,stride=1,bias=False),
                nn.BatchNorm2d(out_channels,eps=1e-3,momentum=1e-2))
            self.downSample = True


    def forward(self,x):
        oneCardinalT = torch.chunk(x,chunks=self.k,dim=-3)
        catList = []
        for i,cardinalM in enumerate(self.cardinalList):
            catList.append(cardinalM(oneCardinalT[i]))
        catTensor = self.conv1(torch.cat(catList,dim=-3))
        if self.downSample:
            addedTensor = catTensor + x.clone()
            return self.DownSample(addedTensor)
        else:
            if self.training and self.drop_conn > 0:
                catTensor = drop_connect_B(catTensor, self.drop_conn)
            return catTensor + x.clone()

class Bottleneck(nn.Module):

    def __init__(self,in_channels,out_channels,layers,pooling = "avg",dropRate = 0.5):
        super().__init__()
        block = list()
        block.append(OneBlock(in_channels, out_channels, pooling=pooling,drop_connect_rate=dropRate))
        for i in range(layers - 1):
            block.append(OneBlock(out_channels, out_channels, pooling=pooling,drop_connect_rate=dropRate))
        self.blocks = nn.Sequential(*block)

    def forward(self,x):
        return self.blocks(x)



class ResNestNet(nn.Module):

    def __init__(self,in_channels,num_classes,dropRate,w,d):
        super().__init__()
        self.conv1 = nn.Sequential(Conv2dDynamicSamePadding(in_channels,64 * w,7,1,bias=False),
                                   nn.BatchNorm2d(64 * w,eps=1e-3,momentum=1e-2),
                                   Mish())
        self.b1 = Bottleneck(64 * w, 128 * w, 3 * d,pooling="AVG",dropRate=dropRate)
        self.b2 = Bottleneck(128 * w, 256 * w, 4 * d,pooling="AVG",dropRate=dropRate)
        self.b3 = Bottleneck(256 * w, 512 * w, 6 * d,pooling="AVG",dropRate=dropRate)
        self.b4 = Bottleneck(512 * w, 1024 * w, 3 * d,pooling="AVG",dropRate=dropRate)
        self.conv2 = nn.Sequential(nn.Conv2d(1024 * w, 1280,1,1,0,bias=False),
                                   nn.BatchNorm2d(1280),
                                   Mish())
        self.dropout = nn.Dropout(dropRate + 0.2 ,True)
        self.fc = nn.Linear(1280 , num_classes)
        self._initialize_weights()

    def forward(self,x):
        conv1 = self.conv1(x)
        #print(conv1.shape)
        b1 = self.b1(conv1)
        #print(b1.shape)
        b2 = self.b2(b1)
        #print(b2.shape)
        b3 = self.b3(b2)
        #print(b3.shape)
        b4 = self.b4(b3)
        #print(b4.shape)
        avgG = F.adaptive_avg_pool2d(self.conv2(b4),[1,1])
        return self.fc(self.dropout(avgG.view(avgG.size()[0],-1)))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



if __name__ == "__main__":
    testInput = torch.randn(size=[5,3,32,32]).float()
    testModule = ResNestNet(3,10,0.5,1,1)
    output = testModule(testInput)
    from Tools import plot_grad_flow
    optimizer = torch.optim.SGD(testModule.parameters(), 5e-4, momentum=0.9, weight_decay=1e-5)
    outputs = testModule(testInput)
    print(outputs)
    lossCri = nn.CrossEntropyLoss(reduction="mean")
    import numpy as np
    loss = lossCri(outputs, torch.from_numpy(np.array([0,1,2,3,4])).long())
    loss.backward()
    # optimizer.zero_grad()
    # optimizer.step()
    # outputs2 = model(testInput)
    # loss = lossCri(outputs2, torch.from_numpy(np.array([0, 1, 2, 3, 4])).long())
    # loss.backward()
    plot_grad_flow(testModule.named_parameters())










