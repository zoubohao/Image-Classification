import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Mish(nn.Module):

    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x) :
        x = x  * (torch.tanh(F.softplus(x)))
        return x

class DepthWiseSeparableConvolution(nn.Module):

    def __init__(self,in_channels,out_channels,splitNumber,kernelSize,padding,groupNumber):
        super(DepthWiseSeparableConvolution,self).__init__()
        self.chunks = splitNumber
        self.conv1 = nn.Conv2d(in_channels,out_channels,1,1,0,bias=True)
        self.gn = nn.GroupNorm(groupNumber,out_channels)
        if out_channels % splitNumber != 0 :
            raise Exception("has error in out channels dividing split number ")
        eachChannels = out_channels // splitNumber
        self.pRelu = Mish()
        self.convs = nn.ModuleList([nn.Conv2d(eachChannels,eachChannels,
                                              kernelSize,1,padding) for _ in range(splitNumber)])

    def forward(self, x):
        ### [B,N,H,W]
        x = self.conv1(x)
        x = self.pRelu(self.gn(x))
        xs = torch.chunk(x,self.chunks,dim=1)
        convedTensors = []
        for i,oneTensor in enumerate(xs):
            convedTensor = self.convs[i](oneTensor)
            convedTensors.append(convedTensor)
        return torch.cat(convedTensors,dim=1)

# class ScaledDotProductAttention(nn.Module):
#
#     def __init__(self,dk):
#         super(ScaledDotProductAttention,self).__init__()
#         self.dk = torch.tensor(dk,requires_grad=False).float()
#
#     def forward(self,Q,K,V):
#         one = torch.div(torch.matmul(Q,torch.transpose(K,0,1)),self.dk)
#         return torch.matmul(torch.softmax(one,1),V)
#
# class OneHeadAttention(nn.Module):
#
#     def __init__(self,dk,in_channels,out_channels):
#         super(OneHeadAttention,self).__init__()
#         self.attention = ScaledDotProductAttention(dk)
#         self.QL = nn.Linear(in_channels,out_channels)
#         self.KL = nn.Linear(in_channels,out_channels)
#         self.VL = nn.Linear(in_channels,out_channels)
#
#
#     def forward(self,x):
#         Ql = self.QL(x)
#         Kl = self.KL(x)
#         Vl = self.VL(x)
#         return self.attention(Ql,Kl,Vl)
#
#
# class MultiHeadAttention(nn.Module):
#
#     def __init__(self,dk,h,in_features):
#         super(MultiHeadAttention,self).__init__()
#         self.h = h
#         self.MultiA = nn.ModuleList([OneHeadAttention(dk,in_features,in_features // 4)
#                                      for _ in range(h)])
#         self.oL = nn.Linear(h * (in_features // 4),in_features)
#         self.act = nn.PReLU()
#
#     def forward(self, x):
#         attenList = []
#         for one in self.MultiA:
#             attenList.append(one(x))
#         return self.act(self.oL(torch.cat(attenList,dim=1)))

class SEBlock(nn.Module):

    def __init__(self,in_features):
        super(SEBlock,self).__init__()
        self.channels = in_features
        self.linear1 = nn.Linear(in_features,in_features // 4)
        self.linear2 = nn.Linear(in_features // 4,in_features)
        self.gPooling = nn.AdaptiveAvgPool2d(output_size=[1,1])
        self.prelu = Mish()
        self.bn = nn.BatchNorm1d(in_features // 4,momentum=0.99)


    def forward(self, x):
        oriX = x.clone()
        x = self.gPooling(x).reshape([-1,self.channels])
        x = self.prelu(self.bn(self.linear1(x)))
        x = self.linear2(x)
        x = torch.sigmoid(x).reshape([-1,self.channels,1,1])
        return torch.mul(oriX,x)

class MBConv(nn.Module):

    def __init__(self,inChannels,kernelSize,padding,groupNumber = 4,splitNumber = 6):
        super(MBConv,self).__init__()
        self.dConv = DepthWiseSeparableConvolution(inChannels,splitNumber * inChannels
                                                   ,splitNumber,kernelSize,padding,groupNumber)
        self.gn1 = nn.GroupNorm(groupNumber,inChannels * splitNumber)
        self.conv1 = nn.Conv2d(splitNumber * inChannels,inChannels,1,1,0)
        self.gn2 = nn.GroupNorm(groupNumber,inChannels)
        self.pRelu1 = Mish()
        self.pRelu2 = Mish()
        self.se = SEBlock(inChannels)

    def forward(self,x):
        OriX = x.clone()
        x = self.dConv(x)
        x = self.pRelu1(self.gn1(x))
        x = self.conv1(x)
        x = self.pRelu2(self.gn2(x))
        se = self.se(x)
        return torch.add(OriX,se)

class  MBconvs(nn.Module) :

    def __init__(self,inChannels,outChannels,layers,kernelSize,padding,strides,splitNumber,dropRate,
                 depth,width):
        super(MBconvs,self).__init__()
        self.depth = depth
        self.width = width
        self.drop = nn.Dropout(dropRate)
        self.convTrans = nn.Conv2d(inChannels * self.width,outChannels * self.width,
                                   3,strides,1)
        self.MBs = nn.ModuleList([MBConv(outChannels * self.width,
                                              kernelSize,padding,splitNumber)
                                       for _ in range(self.depth * layers)])

    def forward(self,x):
        xD = self.drop(x)
        xT = self.convTrans(xD)
        loopTensor = xT.clone()
        for module in self.MBs:
            loopTensor = module(loopTensor)
        return loopTensor


class EfficientNet(nn.Module):

    def __init__(self,imagesChannels, labelsNumber,dropOutRate = 0.2,
                 alpha = 1.2, beta = 1.1, gamma = 1.15,fy = 5):
        super(EfficientNet,self).__init__()
        self.depth = int(math.pow(alpha,fy))
        self.width = int(math.pow(beta,fy))
        self.resolution = int(math.pow(gamma,fy))
        print("Depth ",self.depth)
        print("Width ",self.width)
        print("Resolution ",self.resolution)
        ###
        self.upSample = nn.Upsample(scale_factor=self.resolution,
                                             mode="bicubic",align_corners=True)
        ### 64
        self.conv3x3 = nn.Conv2d(imagesChannels,32 * self.width,
                                  3,1,1)
        ### 64
        self.MBConv1 = MBconvs(inChannels=32,outChannels=16,layers=1,
                               kernelSize=3,padding=1,strides=1,splitNumber=1,
                               dropRate=dropOutRate,
                               depth=self.depth,width=self.width)
        ### 64
        self.MBConv6_1 = MBconvs(inChannels=16,outChannels=24,layers=2,
                               kernelSize=3,padding=1,strides=1,splitNumber=6,
                               dropRate=dropOutRate,
                               depth=self.depth,width=self.width)
        ### 32
        self.MBConv6_2 = MBconvs(inChannels=24,outChannels=42,layers=2,
                               kernelSize=5,padding=2,strides=2,splitNumber=6,
                               dropRate=dropOutRate,
                               depth=self.depth,width=self.width)
        ### 32
        self.MBConv6_3 = MBconvs(inChannels=42,outChannels=84,layers=3,
                               kernelSize=3,padding=1,strides=1,splitNumber=6,
                               dropRate=dropOutRate,
                               depth=self.depth,width=self.width)
        ### 16
        self.MBConv6_4 = MBconvs(inChannels=84,outChannels=114,layers=3,
                               kernelSize=5,padding=2,strides=2,splitNumber=6,
                               dropRate=dropOutRate,
                               depth=self.depth,width=self.width)
        ### 8
        self.MBConv6_5 = MBconvs(inChannels=114,outChannels=192,layers=4,
                               kernelSize=5,padding=2,strides=2,splitNumber=6,
                               dropRate=dropOutRate,
                               depth=self.depth,width=self.width)
        ### 8
        self.MBConv6_6 = MBconvs(inChannels=192,outChannels=324,layers=1,
                               kernelSize=3,padding=1,strides=1,splitNumber=6,
                               dropRate=dropOutRate,
                               depth=self.depth,width=self.width)
        ###
        self.conv1x1 = nn.Conv2d(324 * self.width,1280,1,1,0)
        self.gPooling = nn.AdaptiveAvgPool2d([1,1])
        self.liner = nn.Linear(1280,labelsNumber)

    def forward(self, x):
        upX = self.upSample(x)
        convX = self.conv3x3(upX)
        mb1 = self.MBConv1(convX)
        mb2 = self.MBConv6_1(mb1)
        mb3 = self.MBConv6_2(mb2)
        mb4 = self.MBConv6_3(mb3)
        mb5 = self.MBConv6_4(mb4)
        mb6 = self.MBConv6_5(mb5)
        mb7 = self.MBConv6_6(mb6)
        conv1X = self.conv1x1(mb7)
        pool = self.gPooling(conv1X).reshape([-1,1280])
        return self.liner(pool)



if __name__ == "__main__":
    testInput = torch.randn(size=[8,3,32,32]).float()
    testModule = EfficientNet(3,10)
    print(testModule(testInput))
