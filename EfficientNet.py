import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import OrderedDict

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv2dDynamicSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        h_step = math.ceil(w / self.stride[1])
        v_step = math.ceil(h / self.stride[0])
        h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
        v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)
        extra_h = h_cover_len - w
        extra_v = v_cover_len - h
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        x = F.pad(x, [left, right, top, bottom])
        x = self.conv(x)
        return x

class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        h_step = math.ceil(w / self.stride[1])
        v_step = math.ceil(h / self.stride[0])
        h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
        v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)

        extra_h = h_cover_len - w
        extra_v = v_cover_len - h

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


class MBConvBlock(nn.Module):

    def __init__(self, in_channels,out_channels,kernel_size = 3,dropRate = 0.2,expansion_factor = 6,has_se = True):
        super().__init__()
        bn_mom = 1 - 0.99
        bn_eps = 0.001
        self.dropRate = dropRate
        self.has_se = has_se
        self.expansionConv = Conv2dDynamicSamePadding(in_channels,expansion_factor * in_channels,
                                                      1,1,groups=1)
        self.bn_expansion = nn.BatchNorm2d(in_channels * expansion_factor,
                                           eps=bn_eps,momentum=bn_mom)
        self.dwiseConv = Conv2dDynamicSamePadding(in_channels * expansion_factor,
                                                  in_channels * expansion_factor,kernel_size,
                                                  stride=1,groups=in_channels * expansion_factor,
                                                  bias=False)
        self.point = Conv2dDynamicSamePadding(in_channels * expansion_factor,
                                                  in_channels * expansion_factor,kernel_size=1,stride=1)
        self.bn_Dwise = nn.BatchNorm2d(expansion_factor * in_channels,
                                       bn_eps,momentum=bn_mom)
        self.reduceConv = Conv2dDynamicSamePadding(in_channels * expansion_factor,
                                                   in_channels,1,1)
        self.bn_reduce = nn.BatchNorm2d(in_channels,bn_eps,bn_mom)
        self.blockACT = Swish()
        if self.has_se:
            self._se_reduce = Conv2dDynamicSamePadding(in_channels=expansion_factor * in_channels, out_channels=in_channels, kernel_size=1)
            self._se_expand = Conv2dDynamicSamePadding(in_channels=in_channels, out_channels=expansion_factor * in_channels, kernel_size=1)
        self.residualP = nn.Parameter(torch.ones(size=[1], requires_grad=True), requires_grad=True).float()
        self.oriP = nn.Parameter(torch.ones(size=[1], requires_grad=True), requires_grad=True).float()
        if in_channels == out_channels:
            self.if_down_sample = False
        else:
            self.if_down_sample = True
            self.down_sample_conv = Conv2dDynamicSamePadding(in_channels,out_channels,3,2,bias=True)


    def forward(self, x):
        xOri = x
        xExpansion = self.blockACT(self.bn_expansion(self.expansionConv(x)))
        xDepthWise = self.blockACT(self.bn_Dwise(self.point(self.dwiseConv(xExpansion))))
        xSE = xDepthWise.clone()
        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(xSE, [1,1])
            x_squeezed = self._se_expand(self.blockACT(self._se_reduce(x_squeezed)))
            xSE = torch.sigmoid(x_squeezed) * xSE
        xReduce = self.bn_reduce(self.reduceConv(xSE))
        xGate = xReduce * torch.abs(self.residualP)
        if self.if_down_sample:
            if np.random.rand(1) < self.dropRate:
                return self.down_sample_conv(xOri)
            else:
                return self.down_sample_conv(torch.div(xOri * torch.abs(self.oriP) + xGate,
                                                       torch.abs(self.residualP) + torch.abs(self.oriP) + 1e-6))
        else:
            if np.random.rand(1) < self.dropRate:
                return xOri
            else:
                return torch.div(xOri * torch.abs(self.oriP) + xGate,
                                 torch.abs(self.residualP) + torch.abs(self.oriP) + 1e-6)

class MB_Blocks(nn.Module):

    def __init__(self,in_channels,out_channels,layers,kernel_size = 3,drop_connect_rate = 0.2):
        super(MB_Blocks,self).__init__()
        self.blocks = nn.ModuleList([])
        if layers > 1:
            for _ in range(layers - 1):
                self.blocks.append(MBConvBlock(in_channels, in_channels,kernel_size,drop_connect_rate))
        self.blocks.append(MBConvBlock(in_channels, out_channels, kernel_size, drop_connect_rate))

    def forward(self, x):
        for m in self.blocks:
            x = m(x)
        return x

class CBA(nn.Module):

    def __init__(self,in_channels,outChannels,kernel_size,stride,group):
        super(CBA,self).__init__()
        bn_mom = 1 - 0.99
        bn_eps = 0.001
        self.conv = Conv2dDynamicSamePadding(in_channels,in_channels,kernel_size,stride,groups= group,bias=False)
        self.point = Conv2dDynamicSamePadding(in_channels,outChannels,kernel_size=1,stride=1)
        self.bn = nn.BatchNorm2d(outChannels,bn_eps,bn_mom)
        self.act = Swish()

    def forward(self, x):
        return self.act(self.bn(self.point(self.conv(x))))


class BiFPN_Down_Unit (nn.Module):

    def __init__(self,P_high_channels,P_low_channels,inner_channels,eps = 1e-3):
        super(BiFPN_Down_Unit,self).__init__()
        self.paraHigh = nn.Parameter(torch.ones(size=[1],requires_grad=True),requires_grad=True)
        self.paraLow = nn.Parameter(torch.ones(size=[1],requires_grad=True),requires_grad=True)
        self.eps = eps
        self.convHighTransChannels = CBA(P_high_channels,inner_channels,1,1,1)
        self.convLowTransChannels = CBA(P_low_channels,inner_channels,1,1,1)
        self.innerConvBlocks = CBA(inner_channels,inner_channels,3,1,inner_channels)


    def forward(self,P_high, P_low):

        upSampleHigh = F.interpolate(P_high,size=[P_low.shape[-2],P_low.shape[-1]])
        transHigh = self.convHighTransChannels(upSampleHigh)
        transLow = self.convLowTransChannels(P_low)
        added = torch.div(torch.abs(self.paraHigh) * transHigh + torch.abs(self.paraLow) * transLow,
                          self.eps + torch.abs(self.paraLow) + torch.abs(self.paraHigh))
        return self.innerConvBlocks(added)


class BiFPN_Up_Unit (nn.Module):

    def __init__(self,P_Ori_channels,P_Low_channels,P_Med_channels,inner_channels,eps = 1e-6):
        super(BiFPN_Up_Unit,self).__init__()
        self.paraOri = nn.Parameter(torch.ones(size=[1],requires_grad=True),requires_grad=True)
        self.paraLow = nn.Parameter(torch.ones(size=[1],requires_grad=True),requires_grad=True)
        self.paraMed = nn.Parameter(torch.ones(size=[1], requires_grad=True), requires_grad=True)
        self.eps = eps
        self.convOriTransChannels = CBA(P_Ori_channels,inner_channels,1,1,1)
        self.convLowTransChannels = CBA(P_Low_channels, inner_channels, 1, 1, 1)
        self.convMedTransChannels = CBA(P_Med_channels, inner_channels, 1, 1, 1)
        self.innerConvBlocks = CBA(inner_channels,inner_channels,3,1,inner_channels)
        self.downSampling = MaxPool2dStaticSamePadding(kernel_size=3,stride=2)


    def forward(self,P_Ori,P_Low,P_Med):
        downSample = self.downSampling(P_Low)
        downSample = F.interpolate(downSample,size=[P_Ori.shape[-2],P_Ori.shape[-1]])
        transOri = self.convOriTransChannels(P_Ori)
        transLow = self.convLowTransChannels(downSample)
        transMed = self.convMedTransChannels(P_Med)
        added = torch.div(torch.abs(self.paraLow) * transLow + torch.abs(self.paraMed) + transMed + torch.abs(self.paraOri) * transOri,
                          self.eps + torch.abs(self.paraOri) + torch.abs(self.paraMed) + torch.abs(self.paraLow))
        #added = self.paraLow * transLow + self.paraMed + transMed + self.paraOri * transOri
        return self.innerConvBlocks(added)

class BiFPN(nn.Module):
    """
    illustration of a minimal bifpn unit
        P5_0 -------------------------> P4P5Up -------->
           |-------------|                ↑
                         ↓                |
        P4_0 --------->P5P4Down--------->P3P4Up-------->
           |-------------|--------------↑ ↑
                         ↓                |
        P3_0 --------->P4P3Down--------->P2P3Up-------->
           |-------------|--------------↑ ↑
                         ↓                |
                         |--------------↓ |
        P2_0 -------------------------> P3P2Down-------->
    """

    def __init__(self,P2C,P3C,P4C,P5C,inner_channels):
        super(BiFPN,self).__init__()
        self.P5P4Down = BiFPN_Down_Unit(P5C,P4C,inner_channels)
        self.P4P3Down = BiFPN_Down_Unit(inner_channels,P3C,inner_channels)
        self.P3P2Down = BiFPN_Down_Unit(inner_channels,P2C,inner_channels)
        ##
        self.P2P3Up = BiFPN_Up_Unit(P3C,inner_channels,inner_channels,inner_channels)
        self.P3P4Up = BiFPN_Up_Unit(P4C,inner_channels,inner_channels,inner_channels)
        self.P4P5Up = BiFPN_Down_Unit(inner_channels,P5C,inner_channels)

    def forward(self, P2,P3,P4,P5):
        P4Td = self.P5P4Down(P5,P4)
        P3Td = self.P4P3Down(P4Td,P3)
        P2Out = self.P3P2Down(P3Td,P2)
        ###
        P3Out = self.P2P3Up(P3,P2Out,P3Td)
        P4Out = self.P3P4Up(P4,P3Out,P4Td)
        P5Out = self.P4P5Up(P4Out,P5)
        return P2Out,P3Out,P4Out,P5Out


class EfficientNetReform(nn.Module):

    def __init__(self,in_channels,fy = 1.,drop_connect_rate = 0.2,if_classify = True,num_classes = 10):
        super(EfficientNetReform,self).__init__()
        self.if_classify = if_classify
        bn_mom = 1 - 0.99
        bn_eps = 0.001
        d = math.ceil(math.pow(1.2,fy))  ## depth
        w = math.ceil(math.pow(1.1,fy))  ## width channels
        ### stem
        self.conv_stem = Conv2dDynamicSamePadding(in_channels, 32 * w, kernel_size=3, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=32 * w, momentum=bn_mom, eps=bn_eps)
        ### blocks
        ### r1
        self.block1 = MB_Blocks(32 * w, 24 * w, layers=1 * d, kernel_size=3, drop_connect_rate=drop_connect_rate)
        self.block2 = MB_Blocks(24 * w, 24 * w, layers=2 * d, kernel_size=3, drop_connect_rate=drop_connect_rate)
        ### r2
        self.block3 = MB_Blocks(24 * w, 40 * w, layers=2 * d, kernel_size=5, drop_connect_rate=drop_connect_rate)
        ### r3
        self.block4 = MB_Blocks(40 * w, 80 * w, layers=3 * d, kernel_size=3, drop_connect_rate=drop_connect_rate)
        ### r4
        self.block5 = MB_Blocks(80 * w, 192 * w, layers=3 * d, kernel_size=5, drop_connect_rate=drop_connect_rate)
        self.block6 = MB_Blocks(192 * w, 192 * w, layers=4 * d, kernel_size=5, drop_connect_rate=drop_connect_rate)
        ### r5
        self.block7 = MB_Blocks(192 * w, 320 * w, layers=1 * d, kernel_size=3, drop_connect_rate=drop_connect_rate)
        ### BiFPN
        inner_channels = 64 * math.ceil(math.pow(1.35,fy))
        layers = 2 + math.ceil(math.pow(1.35,fy))
        self.BiFPN1 = BiFPN(40 * w,80 * w,192 * w, 320 * w,inner_channels)
        self.BiFPNList = nn.ModuleList([BiFPN(inner_channels,inner_channels,inner_channels,inner_channels,inner_channels) for _ in range(int(layers))])
        ### classify
        if if_classify:
            self.paras = nn.ParameterList([nn.Parameter(torch.ones(size=[1],requires_grad=True).float(),requires_grad=True) for _ in range(4)])
            self.finalConv = CBA(inner_channels,1280,1,1,1)
            self.linear = nn.Linear(1280,num_classes,bias=False)


    def forward(self,x):
        """
        :param x:
        :return:
        """
        #print(x.shape)
        xStem = self.bn0(self.conv_stem(x))
        p1 = self.block2(self.block1(xStem))
        p2 = self.block3(p1)
        p3 = self.block4(p2)
        p4 = self.block6(self.block5(p3))
        p5 = self.block7(p4)
        B12,B13,B14,B15 = self.BiFPN1(p2,p3,p4,p5)
        for m in self.BiFPNList:
            B12, B13, B14, B15 = m(B12,B13,B14,B15)
        if self.if_classify:
            hs,ws = B15.shape[-2],B15.shape[-1]
            #print(B15.shape)
            B32D = F.interpolate(B12, size=[hs, ws])
            B33D = F.interpolate(B13, size=[hs, ws])
            B34D = F.interpolate(B14, size=[hs, ws])
            added = torch.abs(self.paras[0]) * B32D + torch.abs(self.paras[1]) * B33D + torch.abs(self.paras[2]) * B34D + torch.abs(self.paras[3]) * B15
            #print(added.shape)
            norm = torch.div(added,torch.abs(self.paras[0]) + torch.abs(self.paras[1]) + torch.abs(self.paras[2]) + torch.abs(self.paras[3]) + 0.0001)
            avgTensor = F.adaptive_avg_pool2d(self.finalConv(norm), output_size=[1, 1])
            return self.linear(torch.squeeze(torch.squeeze(avgTensor,-1),-1))
        else:
            return OrderedDict([("0",B15),("1",B14),("2",B13),("3",B12)])


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
            layers.append(n.split(".")[-1])
            print("########")
            print(n.split(".")[0:-2])
            if p.grad is not None:
                print(p.grad.abs().mean())
                if p.grad.abs().mean() > 2:
                    ave_grads.append(2)
                else:
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
    # testMBconv = MB_Blocks(32 ,16,layers=1)
    # testInput = torch.randn(size=[5,32 ,32,32]).float()
    # print(testMBconv(testInput).shape)
    # print(math.ceil(math.pow(1.2,2)))
    # #########
    # testP_High = torch.randn(size=[5,32,7,7]).float()
    # testP_Low = torch.randn(size=[5,16,15,15]).float()
    # testBi = BiFPN_Down_Unit(32,16,inner_channels=64,layers=2)
    # print(testBi(testP_High,testP_Low).shape)
    # ##########
    # testP5 = torch.randn(size=[5,16,30,30]).float()
    # testP6Oir = torch.randn(size=[5,64,16,16]).float()
    # testP6Med = torch.randn(size=[5,32,16,16]).float()
    # testUp = BiFPN_Up_Unit(64,16,32,64,3)
    # print(testUp(testP6Oir,testP5,testP6Med).shape)
    # ##########
    # testP2 = torch.randn(size=[5,32,64,64]).float()
    # testP3 = torch.randn(size=[5, 64, 30, 30]).float()
    # testP4 = torch.randn(size=[5, 128, 17, 17]).float()
    # testP5 = torch.randn(size=[5, 256, 7, 7]).float()
    # testBiFPN = BiFPN(32,64,128,256,64,2)
    # r2,r3,r4,r5 = testBiFPN(testP2,testP3,testP4,testP5)
    # print(r2.shape)
    # print(r3.shape)
    # print(r4.shape)
    # print(r5.shape)
    ##########
    testInput = torch.randn(size=[5,3,32,32]).float()
    model = EfficientNetReform(in_channels=3,fy=4,if_classify=True)
    outputs = model(testInput)
    print(outputs)
    lossCri = nn.CrossEntropyLoss(reduction="sum")
    loss = lossCri(outputs, torch.from_numpy(np.array([0,1,2,3,4])).long())
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
    plot_grad_flow(model.named_parameters())
    # print(finalDic["1"].shape)
    # print(finalDic["2"].shape)
    # print(finalDic["3"].shape)

