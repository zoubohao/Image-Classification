import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import OrderedDict

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MBConvBlock(nn.Module):

    def __init__(self, in_channels,out_channels,kernel_size = 3,dropRate = 0.2,expansion_factor = 6,has_se = True):
        super().__init__()
        bn_mom = 1 - 0.99
        bn_eps = 0.001
        self.dropRate = dropRate
        self.has_se = has_se
        self.expansionConv = Conv2dDynamicSamePadding(in_channels,expansion_factor * in_channels,
                                                      1,1,groups=1,bias=False)
        self.bn_expansion = nn.BatchNorm2d(in_channels * expansion_factor,
                                           eps=bn_eps,momentum=bn_mom)
        self.dwiseConv = Conv2dDynamicSamePadding(in_channels * expansion_factor,
                                                  in_channels * expansion_factor,kernel_size,
                                                  stride=1,groups=in_channels * expansion_factor,
                                                  bias=False)
        self.bn_Dwise = nn.BatchNorm2d(expansion_factor * in_channels,
                                       bn_eps,momentum=bn_mom)
        self.reduceConv = Conv2dDynamicSamePadding(in_channels * expansion_factor,
                                                   in_channels,1,1,bias=False)
        self.bn_reduce = nn.BatchNorm2d(in_channels,bn_eps,bn_mom)
        self.blockACT = Swish()
        if self.has_se:
            self._se_reduce = Conv2dDynamicSamePadding(in_channels=expansion_factor * in_channels, out_channels=in_channels, kernel_size=1)
            self._se_expand = Conv2dDynamicSamePadding(in_channels=in_channels, out_channels=expansion_factor * in_channels, kernel_size=1)
        self.residualP = nn.Parameter(torch.zeros(size=[1],requires_grad=True),requires_grad=True).float()
        if in_channels == out_channels:
            self.if_down_sample = False
        else:
            self.if_down_sample = True
            self.down_sample_conv = Conv2dDynamicSamePadding(in_channels,out_channels,3,2,bias=True)


    def forward(self, x):
        xOri = x.clone()
        xExpansion = self.blockACT(self.bn_expansion(self.expansionConv(x)))
        xDepthWise = self.blockACT(self.bn_Dwise(self.dwiseConv(xExpansion)))
        xSE = xDepthWise.clone()
        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(xSE, [1,1])
            x_squeezed = self._se_expand(self.blockACT(self._se_reduce(x_squeezed)))
            xSE = torch.sigmoid(x_squeezed) * xSE
        xReduce = self.blockACT(self.bn_reduce(self.reduceConv(xSE)))
        xGate = xReduce * self.residualP
        if self.if_down_sample:
            if np.random.rand(1) < self.dropRate:
                return self.down_sample_conv(xOri)
            else:
                return self.down_sample_conv(xOri + xGate)
        else:
            if np.random.rand(1) < self.dropRate:
                return xOri
            else:
                return xOri + xGate

class MB_Blocks(nn.Module):

    def __init__(self,in_channels,out_channels,layers,kernel_size = 3,drop_connect_rate = 0.2):
        super(MB_Blocks,self).__init__()
        self.blocks = nn.ModuleList([])
        self.blocks.append(MBConvBlock(in_channels,out_channels,kernel_size,drop_connect_rate))
        if layers > 1:
            for _ in range(layers - 1):
                self.blocks.append(MBConvBlock(out_channels, out_channels,kernel_size,drop_connect_rate))

    def forward(self, x):
        for m in self.blocks:
            x = m(x)
        return x

class CBA(nn.Module):

    def __init__(self,in_channels,outChannels,kernel_size,stride,group):
        super(CBA,self).__init__()
        bn_mom = 1 - 0.99
        bn_eps = 0.001
        self.conv = Conv2dDynamicSamePadding(in_channels,outChannels,kernel_size,stride,groups= group)
        self.bn = nn.BatchNorm2d(outChannels,bn_eps,bn_mom)
        self.act = Swish()
        self.if_res = False
        if in_channels == outChannels :
            self.p = nn.Parameter(torch.zeros(size=[1], requires_grad=True), requires_grad=True)
            self.if_res = True
    def forward(self, x):
        if self.if_res:
            return self.act(self.bn(self.conv(x))) * self.p + x
        else:
            return self.act(self.bn(self.conv(x)))

class BiFPN_Down_Unit (nn.Module):

    def __init__(self,P_high_channels,P_low_channels,inner_channels, layers,eps = 1e-3):
        super(BiFPN_Down_Unit,self).__init__()
        self.paraHigh = nn.Parameter(torch.zeros(size=[1],requires_grad=True),requires_grad=True)
        self.paraLow = nn.Parameter(torch.zeros(size=[1],requires_grad=True),requires_grad=True)
        self.eps = eps
        self.convHighTransChannels = CBA(P_high_channels,inner_channels,1,1,1)
        self.convLowTransChannels = CBA(P_low_channels,inner_channels,1,1,1)
        self.innerConvBlocks = nn.ModuleList([CBA(inner_channels,inner_channels,3,1,inner_channels) for _ in range(layers)])


    def forward(self,P_high, P_low):

        upSampleHigh = F.interpolate(P_high,size=[P_low.shape[-2],P_low.shape[-1]])
        transHigh = self.convHighTransChannels(upSampleHigh)
        transLow = self.convLowTransChannels(P_low)
        addedHighLow = torch.div(self.paraHigh * transHigh + self.paraLow * transLow,self.eps + self.paraLow + self.paraHigh)
        for m in self.innerConvBlocks:
            addedHighLow = m(addedHighLow)
        return addedHighLow


class BiFPN_Up_Unit (nn.Module):

    def __init__(self,P_Ori_channels,P_Low_channels,P_Med_channels,inner_channels, layers,eps = 1e-3):
        super(BiFPN_Up_Unit,self).__init__()
        self.paraOri = nn.Parameter(torch.zeros(size=[1],requires_grad=True),requires_grad=True)
        self.paraLow = nn.Parameter(torch.zeros(size=[1],requires_grad=True),requires_grad=True)
        self.paraMed = nn.Parameter(torch.zeros(size=[1], requires_grad=True), requires_grad=True)
        self.eps = eps
        self.convOriTransChannels = CBA(P_Ori_channels,inner_channels,1,1,1)
        self.convLowTransChannels = CBA(P_Low_channels, inner_channels, 1, 1, 1)
        self.convMedTransChannels = CBA(P_Med_channels, inner_channels, 1, 1, 1)
        self.innerConvBlocks = nn.ModuleList([CBA(inner_channels,inner_channels,3,1,inner_channels) for _ in range(layers)])


    def forward(self,P_Ori,P_Low,P_Med):
        downSample = F.interpolate(P_Low,size=[P_Ori.shape[-2],P_Ori.shape[-1]])
        transOri = self.convOriTransChannels(P_Ori)
        transLow = self.convLowTransChannels(downSample)
        transMed = self.convMedTransChannels(P_Med)
        added = torch.div(self.paraLow * transLow + self.paraMed + transMed + self.paraOri * transOri,
                          self.eps + self.paraOri + self.paraMed + self.paraLow)
        for m in self.innerConvBlocks:
            added = m(added)
        return added

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

    def __init__(self,P2C,P3C,P4C,P5C,inner_channels, layers):
        super(BiFPN,self).__init__()
        self.P5P4Down = BiFPN_Down_Unit(P5C,P4C,inner_channels,layers)
        self.P4P3Down = BiFPN_Down_Unit(inner_channels,P3C,inner_channels,layers)
        self.P3P2Down = BiFPN_Down_Unit(inner_channels,P2C,inner_channels,layers)
        ##
        self.P2P3Up = BiFPN_Up_Unit(P3C,inner_channels,inner_channels,inner_channels,layers)
        self.P3P4Up = BiFPN_Up_Unit(P4C,inner_channels,inner_channels,inner_channels,layers)
        self.P4P5Up = BiFPN_Down_Unit(inner_channels,P5C,inner_channels,layers)

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
        self.r = 64 * math.ceil(math.pow(1.15,fy)) ## resolution
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
        layers = 3 + fy
        self.BiFPN1 = BiFPN(40 * w,80 * w,192 * w, 320 * w,inner_channels,layers)
        self.BiFPN2 = BiFPN(inner_channels,inner_channels,inner_channels, inner_channels,inner_channels,layers)
        self.BiFPN3 = BiFPN(inner_channels, inner_channels, inner_channels, inner_channels, inner_channels, layers)
        ### classify
        if if_classify:
            self.paras = nn.ParameterList([nn.Parameter(torch.zeros(size=[1],requires_grad=True).float(),requires_grad=True) for _ in range(4)])
            self.finalConv = CBA(inner_channels,1280,1,1,1)
            self.dropout = nn.Dropout(p = drop_connect_rate)
            self.linear = nn.Linear(1280,num_classes,bias=True)


    def forward(self,x):
        """
        :param x:
        :return:
        """
        h , w = x.shape[-2],x.shape[-1]
        xReshape = F.interpolate(x,size=[h + self.r , w + self.r],mode="bilinear",align_corners=True)
        xStem = self.bn0(self.conv_stem(xReshape))
        p1 = self.block2(self.block1(xStem))
        p2 = self.block3(p1)
        p3 = self.block4(p2)
        p4 = self.block6(self.block5(p3))
        p5 = self.block7(p4)
        B12,B13,B14,B15 = self.BiFPN1(p2,p3,p4,p5)
        B22, B23, B24, B25 = self.BiFPN2(B12,B13,B14,B15)
        B32, B33, B34, B35 = self.BiFPN3(B22, B23, B24, B25)
        if self.if_classify:
            hs,ws = B35.shape[-2],B35.shape[-1]
            B32D = F.interpolate(B32, size=[hs, ws])
            B33D = F.interpolate(B33, size=[hs, ws])
            B34D = F.interpolate(B34, size=[hs, ws])
            added = self.paras[0] * B32D + self.paras[1] * B33D + self.paras[2] * B34D + self.paras[3] * B35
            norm = torch.div(added,self.paras[0] + self.paras[1] +self.paras[2] +self.paras[3]+0.0001)
            avgTensor = F.adaptive_avg_pool2d(self.finalConv(norm), output_size=[1, 1])
            return self.linear(self.dropout(torch.squeeze(torch.squeeze(avgTensor,-1),-1)))
        else:
            return OrderedDict([("0",B32),("1",B32),("2",B34),("3",B35)])


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
    model = EfficientNetReform(in_channels=3,fy=1)
    finalDic = model(testInput)
    print(finalDic)

