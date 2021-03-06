import torch
import torch.nn as nn
import torch.nn.functional as F


class FilterResponseNormalization(nn.Module):

    def __init__(self,C):
        super(FilterResponseNormalization,self).__init__()
        self.C = C
        self.eps = nn.Parameter(data=torch.tensor(1e-6),
                                requires_grad=True)
        self.gamma = nn.Parameter(data=torch.ones(size=[C],dtype=torch.float32,requires_grad=True),
                                  requires_grad=True)
        self.beta = nn.Parameter(data=torch.zeros(size=[C],dtype=torch.float32,requires_grad=True),
                                 requires_grad=True)
        self.tao = nn.Parameter(data=torch.zeros(size=[C],dtype=torch.float32,requires_grad=True),
                                requires_grad=True)

    def forward(self, x):
        """
        :param x: shape [N,C,H,W]
        :return: The result of this layer.
        """
        #print(self.eps.shape)
        meanHW = torch.mean(x.pow(2), dim=[2, 3], keepdim=True) + torch.abs(self.eps)
        #print(meanHW.shape)
        normT = x * torch.rsqrt(meanHW)
        shiftT = torch.mul(normT,self.gamma.view([-1,self.C,1,1])) + self.beta.view([-1,self.C,1,1])
        return torch.max(shiftT,self.tao.view([-1,self.C,1,1]))

class SE_Block(nn.Module):

    def __init__(self,inChannels : int):
        super(SE_Block,self).__init__()
        self.linear1 = nn.Linear(inChannels,inChannels // 4)
        self.linear2 = nn.Linear(inChannels // 4, inChannels)
        self.activation = nn.PReLU(init=0.)

    def forward(self,x) :
        shapeX = x.shape
        twoDimensionsT = F.adaptive_avg_pool2d(x,output_size=(1,1)).reshape([shapeX[0],shapeX[1]])
        linear1T = self.linear1(twoDimensionsT)
        act1T = self.activation(linear1T)
        linear2T = self.linear2(act1T)
        sigmoidT = torch.sigmoid(linear2T).unsqueeze(-1).unsqueeze(-1)
        return x + torch.mul(sigmoidT,x)


class BasicBlock (nn.Module):
    """
    This block contains
    1x1 convolution,
    3x3 convolution,
    Mish activation,
    FR normalization,
    B normalization,
    """
    def __init__(self,inChannels : int,growthRate : int,ifUseBn = True):
        super(BasicBlock,self).__init__()
        self.conv1x1_1 = nn.Conv2d(inChannels,growthRate,kernel_size=(1,1),
                                   stride=(1,1),padding=(0,0),bias=True)
        self.conv3x3_1 = nn.Conv2d(growthRate,growthRate,kernel_size=(3,3),
                                   stride=(1,1),padding=(1,1),bias=True)
        self.activation1 = nn.PReLU(inChannels,init=0.)
        self.activation2 = nn.PReLU(growthRate,init=0.)
        if ifUseBn:
            self.normList = nn.ModuleList([nn.BatchNorm2d(inChannels) , nn.BatchNorm2d(growthRate)])
        else:
            self.normList = nn.ModuleList([FilterResponseNormalization(inChannels),FilterResponseNormalization(growthRate)])

    def forward(self,x):
        ### 1
        norm1 = self.normList[0](x)
        act1 = self.activation1(norm1)
        conv1T = self.conv1x1_1(act1)
        ### 2
        norm2T = self.normList[1](conv1T)
        act2 = self.activation2(norm2T)
        conv2T = self.conv3x3_1(act2)
        return conv2T


class UpChannelsAndDownSample(nn.Module):

    def __init__(self,inChannels : int,outChannels:int,pooling= "Avg"):
        super(UpChannelsAndDownSample,self).__init__()
        self.conv = nn.Conv2d(inChannels,outChannels,kernel_size=3,stride=1,padding=1)
        if pooling.lower() == "avg":
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pooling(conv)
        #print(pool.shape)
        return pool


class UpBlocks(nn.Module):

    def __init__(self,inChannels,outChannels,growthRate,blocksNumber,ifUseBn=True,pooling= "Avg"):
        super(UpBlocks,self).__init__()
        self.layerList = nn.ModuleList([BasicBlock(inChannels + b * growthRate ,growthRate,ifUseBn) for b in range(blocksNumber)])
        self.upCha = UpChannelsAndDownSample(inChannels + blocksNumber * growthRate,outChannels,pooling)
        if ifUseBn:
            self.norm = nn.BatchNorm2d(outChannels)
        else:
            self.norm = FilterResponseNormalization(outChannels)
        self.seBlock = SE_Block(outChannels)

    def forward(self, x):
        x1s = [x]
        for currentM in self.layerList:
            tempResult = currentM(torch.cat(x1s,dim=-3))
            x1s.append(tempResult)
        x1O = torch.cat(x1s,dim=-3)
        normT = self.norm(self.upCha(x1O))
        return self.seBlock(normT)


class MyModel (nn.Module):

    def __init__(self,ImageChannels : int,labelsNumber : int,blocks: list,growthRate : int,ifUseBn = True):
        super(MyModel,self).__init__()
        ### layer1 32
        self.upCha1 = nn.Conv2d(ImageChannels,64,kernel_size=7,stride=1,padding=3)
        self.layer1 = UpBlocks(64, 128, growthRate,blocks[0],ifUseBn,"Max")
        ### layer2 16
        self.layer2 = UpBlocks(128,256,growthRate,blocks[1] ,ifUseBn,"Max")
        ### layer3 8
        self.layer3= UpBlocks(256,512,growthRate, blocks[2],ifUseBn)
        ### layer4 4
        self.layer4 = UpBlocks(512,1024,growthRate, blocks[3],ifUseBn)
        ### bottom
        self.gPooling = nn.AdaptiveAvgPool2d(1)
        self.liner1 = nn.Linear(1024,labelsNumber)

    def forward(self,x):
        x = self.upCha1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        gT = self.gPooling(x).reshape([-1,1024])
        liner1 = self.liner1(gT)
        return liner1 , gT



if __name__ == "__main__":
    testInput = torch.randn([16,3,32,32]).float()
    testModel = MyModel(3,10,[6,12,24,16],32,False)
    print(testModel)
    print(testModel(testInput))











