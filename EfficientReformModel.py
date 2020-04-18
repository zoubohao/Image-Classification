import torch
import torch.nn as nn
import torch.nn.functional as F
from Tools import Conv2dDynamicSamePadding
from collections import OrderedDict
from BiFPN import BiFPN
from Tools import Pool2dStaticSamePadding
from Tools import Swish
from SplitAttention import ResNeSt

class MBConvBlock(nn.Module):

    def __init__(self, in_channels,out_channels,dropRate = 0.2,expansion_factor = 2):
        super().__init__()
        ### expansion block
        self.expansionConv = Conv2dDynamicSamePadding(in_channels,expansion_factor * in_channels,1,1,groups=1,bias=False)
        self.bn_expansion = nn.GroupNorm(num_groups=8,num_channels=in_channels * expansion_factor,eps=0.001)
        ### resNeSt block
        self.resNeSt = ResNeSt(k = 4,r = 4,in_channels=expansion_factor * in_channels)
        self.bn_Dwise = nn.GroupNorm(8,expansion_factor * in_channels,0.001)
        ### reduce block
        self.reduceConv = Conv2dDynamicSamePadding(in_channels * expansion_factor,in_channels,1,1,bias=False)
        self.bn_reduce = nn.GroupNorm(4,in_channels,0.001)
        ###
        self.actLayer = Swish()
        if in_channels == out_channels:
            self.if_down_sample = False
        else:
            self.if_down_sample = True
            self.dropOut = nn.Dropout2d(p=dropRate)
            self.down_sample_conv = nn.Sequential(Conv2dDynamicSamePadding(in_channels,out_channels,1,1),
                                                  Pool2dStaticSamePadding(3,2))


    def forward(self, x):
        xOri = x.clone()
        xExpansion = self.actLayer(self.bn_expansion(self.expansionConv(x)))
        xResNeSt = self.actLayer(self.bn_Dwise(self.resNeSt(xExpansion)))
        xReduce = self.bn_reduce(self.reduceConv(xResNeSt))
        if self.if_down_sample:
            return self.dropOut(self.down_sample_conv(xReduce + xOri))
        else:
            return  xReduce + xOri

class MB_Blocks(nn.Module):

    def __init__(self,in_channels,out_channels,layers,drop_connect_rate = 0.2):
        super(MB_Blocks,self).__init__()
        blocksDic = OrderedDict()
        for i in range(layers):
            blocksDic[str(i)] = MBConvBlock(in_channels, in_channels,drop_connect_rate)
        self.seq = nn.Sequential(blocksDic)
        self.trans = MBConvBlock(in_channels, out_channels, drop_connect_rate)

    def forward(self, x):
        return self.trans(self.seq(x))


class EfficientNetReform(nn.Module):

    def __init__(self,in_channels,w = 3,d = 3,drop_connect_rate = 0.2,num_classes = 10,classify = True):
        super(EfficientNetReform,self).__init__()
        ### stem 2X 32
        self.conv_stem = Conv2dDynamicSamePadding(in_channels, 32 * w, kernel_size=7, stride=2)
        self.bn0 = nn.GroupNorm(8,num_channels=32 * w,  eps=0.001)
        ### blocks
        ### r1 2X 16
        self.block1 = MB_Blocks(32 * w, 32 * w, layers=1 * d,  drop_connect_rate=drop_connect_rate)
        self.block2 = MB_Blocks(32 * w, 40 * w, layers=2 * d,  drop_connect_rate=drop_connect_rate)
        ### r2 2X 8
        self.block3 = MB_Blocks(40 * w, 40 * w, layers=2 * d,  drop_connect_rate=drop_connect_rate)
        self.block4 = MB_Blocks(40 * w, 80 * w, layers=3 * d,  drop_connect_rate=drop_connect_rate)
        ### r3 2X 4
        self.block5 = MB_Blocks(80 * w, 80 * w, layers=3 * d, drop_connect_rate=drop_connect_rate)
        self.block6 = MB_Blocks(80 * w, 160 * w, layers=4 * d, drop_connect_rate=drop_connect_rate)
        ### r4 2X 2
        self.block7 = MB_Blocks(160 * w, 320 * w, layers=3 * d,  drop_connect_rate=drop_connect_rate)
        self.block8 = MB_Blocks(320 * w, 320 * w, layers=2 * d, drop_connect_rate=drop_connect_rate)
        ### BiFPN 8, 4, 2
        self.BifpnFirst = BiFPN(num_channels=128 * w // 2, conv_channels=[80 * w,160 * w,320 * w],first_time=True)
        self.Bifpn = BiFPN(128 * w // 2,conv_channels=[],first_time=False)
        self.actLayer = Swish()
        ### classify
        self.classify = classify
        if classify :
            self.seq = nn.Sequential(nn.Linear(128 * w // 2,1280),
                                     nn.BatchNorm1d(1280),
                                     Swish(),
                                     nn.Dropout(drop_connect_rate),
                                     nn.Linear(1280,num_classes))


    def forward(self,x):
        """
        :param x:
        :return:
        """
        #print(x.shape)
        xStem = self.actLayer(self.bn0(self.conv_stem(x)))
        p1 = self.block2(self.block1(xStem))
        p2 = self.block4(self.block3(p1))
        p3 = self.block6(self.block5(p2))
        p4 = self.block8(self.block7(p3))
        rP3, rP4, rP5 = self.BifpnFirst(p2,p3,p4)
        rP3, rP4, rP5 = self.Bifpn(rP3, rP4, rP5)
        if self.classify:
            feat1 = F.adaptive_avg_pool2d(rP3,output_size=[1,1])
            feat2 = F.adaptive_avg_pool2d(rP4,output_size=[1,1])
            feat3 = F.adaptive_avg_pool2d(rP5,output_size=[1,1])
            featFinal = feat1 + feat2 + feat3
            return self.seq(torch.squeeze(torch.squeeze(featFinal, -1), -1))
        else:
            return OrderedDict([("0",rP5),("1",rP4),("2",rP3)])


import matplotlib.pyplot as plt
### Check grad
def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            print("########")
            print(n)
            if p.grad is not None:
                print(p.grad.abs().mean())
                ave_grads.append(p.grad.abs().mean())
            else:
                print(0)
                ave_grads.append(0)
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    from torch.optim import rmsprop
    testInput = torch.randn(size=[5,3,32,32 ]).float()
    model = EfficientNetReform(in_channels=3,w=1,d=1)
    optimizer = rmsprop.RMSprop(model.parameters(), 5e-4, momentum=0.9, weight_decay=1e-5)
    outputs = model(testInput)
    print(outputs)
    lossCri = nn.CrossEntropyLoss(reduction="sum")
    import numpy as np
    loss = lossCri(outputs, torch.from_numpy(np.array([0,1,2,3,4])).long())
    loss.backward()
    # optimizer.zero_grad()
    # optimizer.step()
    # outputs2 = model(testInput)
    # loss = lossCri(outputs2, torch.from_numpy(np.array([0, 1, 2, 3, 4])).long())
    # loss.backward()
    plot_grad_flow(model.named_parameters())

