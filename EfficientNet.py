import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from BiFPN import BiFPN

def Swish(x):
    return x * torch.sigmoid(x)

def AddN(tensorList : []):
    if len(tensorList)==1:
        return tensorList[0]
    else:
        addR = tensorList[0] + tensorList[1]
        for i in range(2,len(tensorList)):
            addR = addR + tensorList[i]
        return addR

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


class MBConvBlock(nn.Module):

    def __init__(self, in_channels,out_channels,kernel_size = 3,dropRate = 0.2,expansion_factor = 3):
        super().__init__()
        self.expansionConv = Conv2dDynamicSamePadding(in_channels,expansion_factor * in_channels,1,1,groups=1,bias=False)
        self.bn_expansion = nn.BatchNorm2d(in_channels * expansion_factor,eps=0.001,momentum=0.01)
        ###
        self.dwiseConv = Conv2dDynamicSamePadding(in_channels * expansion_factor,
                                                  in_channels * expansion_factor,kernel_size,
                                                  stride=1,groups=in_channels * expansion_factor,bias=False)
        self.point = Conv2dDynamicSamePadding(in_channels * expansion_factor,in_channels * expansion_factor,kernel_size=1,stride=1)
        self.bn_Dwise = nn.BatchNorm2d(expansion_factor * in_channels,0.001,momentum=0.01)
        ###
        self.reduceConv = Conv2dDynamicSamePadding(in_channels * expansion_factor,in_channels,1,1,bias=False)
        self.bn_reduce = nn.BatchNorm2d(in_channels,0.001,0.01)
        ###
        self._se_reduce = Conv2dDynamicSamePadding(in_channels=expansion_factor * in_channels, out_channels=in_channels, kernel_size=1,bias=False)
        self._se_expand = Conv2dDynamicSamePadding(in_channels=in_channels, out_channels=expansion_factor * in_channels, kernel_size=1,bias=False)
        if in_channels == out_channels:
            self.if_down_sample = False
        else:
            self.if_down_sample = True
            self.dropOut = nn.Dropout2d(p=dropRate)
            self.down_sample_conv = Conv2dDynamicSamePadding(in_channels,out_channels,3,2,bias=False)


    def forward(self, x):
        xOri = x.clone()
        xExpansion = Swish(self.bn_expansion(self.expansionConv(x)))
        xDepthWise = Swish(self.bn_Dwise(self.point(self.dwiseConv(xExpansion))))
        xDc = xDepthWise.clone()
        # Squeeze and Excitation
        x_squeezed = F.adaptive_avg_pool2d(xDc, [1,1])
        x_squeezed = self._se_expand(Swish(self._se_reduce(x_squeezed)))
        xSE = torch.sigmoid(x_squeezed) * xDepthWise
        xReduce = Swish(self.bn_reduce(self.reduceConv(xSE)))
        if self.if_down_sample:
            return self.dropOut(self.down_sample_conv(xReduce + xOri))
        else:
            return  xReduce + xOri

class MB_Blocks(nn.Module):

    def __init__(self,in_channels,out_channels,layers,kernel_size = 3,drop_connect_rate = 0.2):
        super(MB_Blocks,self).__init__()
        self.blocks = nn.ModuleList([MBConvBlock(in_channels, in_channels,kernel_size,drop_connect_rate) for _ in range(layers)])
        self.trans = MBConvBlock(in_channels, out_channels, kernel_size, drop_connect_rate)

    def forward(self, x):
        for m in self.blocks:
            x = m(x)
        return self.trans(x)



class EfficientNetReform(nn.Module):

    def __init__(self,in_channels,w = 3,d = 3,drop_connect_rate = 0.2,num_classes = 10,classify = True):
        super(EfficientNetReform,self).__init__()
        ### stem
        self.conv_stem = Conv2dDynamicSamePadding(in_channels, 32 * w, kernel_size=3, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=32 * w, momentum=0.001, eps=0.001)
        ### blocks
        ### r1
        self.block1 = MB_Blocks(32 * w, 32 * w, layers=1 * d, kernel_size=3, drop_connect_rate=drop_connect_rate)
        self.block2 = MB_Blocks(32 * w, 32 * w, layers=2 * d, kernel_size=3, drop_connect_rate=drop_connect_rate)
        self.block3 = MB_Blocks(32 * w, 32 * w, layers=2 * d, kernel_size=5, drop_connect_rate=drop_connect_rate)
        self.block4 = MB_Blocks(32 * w, 80 * w, layers=3 * d, kernel_size=3, drop_connect_rate=drop_connect_rate)
        ### r2
        self.block5 = MB_Blocks(80 * w, 192 * w, layers=3 * d, kernel_size=5, drop_connect_rate=drop_connect_rate)
        self.block6 = MB_Blocks(192 * w, 192 * w, layers=4 * d, kernel_size=5, drop_connect_rate=drop_connect_rate)
        ### r3
        self.block7 = MB_Blocks(192 * w, 320 * w, layers=1 * d, kernel_size=3, drop_connect_rate=drop_connect_rate)
        self.block8 = MB_Blocks(320 * w, 320 * w, layers=1 * d, kernel_size=3, drop_connect_rate=drop_connect_rate)
        ### BiFPN
        self.BifpnFirst = BiFPN(num_channels=64 * 2, conv_channels=[80 * w,192 * w,320 * w],first_time=True)
        self.Bifpn = BiFPN(64  * 2,conv_channels=[],first_time=False)
        ### classify
        self.classify = classify
        if classify :
            self.linearTrans = nn.Linear(64 * 2, 1280, bias=True)
            self.bnL = nn.BatchNorm1d(num_features=1280, eps=1e-3, momentum=0.01)
            self.linear = nn.Linear(1280, num_classes, bias=False)


    def forward(self,x):
        """
        :param x:
        :return:
        """
        #print(x.shape)
        xStem = Swish(self.bn0(self.conv_stem(x)))
        p1 = self.block4(self.block3(self.block2(self.block1(xStem))))
        p2 = self.block6(self.block5(p1))
        p3 = self.block8(self.block7(p2))
        rP3, rP4, rP5, rP6, rP7 = self.BifpnFirst(p1,p2,p3)
        rP3, rP4, rP5, rP6, rP7 = self.Bifpn(rP3, rP4, rP5, rP6, rP7)
        if self.classify:
            feat1 = F.adaptive_avg_pool2d(rP3,output_size=[1,1])
            feat2 = F.adaptive_avg_pool2d(rP4,output_size=[1,1])
            feat3 = F.adaptive_avg_pool2d(rP5,output_size=[1,1])
            feat4 = F.adaptive_avg_pool2d(rP6,output_size=[1,1])
            feat5 = F.adaptive_avg_pool2d(rP7,output_size=[1,1])
            featFinal = feat1 + feat2 + feat3 + feat4 + feat5
            return self.linear(Swish(self.bnL(self.linearTrans(torch.squeeze(torch.squeeze(F.adaptive_avg_pool2d(featFinal, [1, 1]), -1), -1)))))
        else:
            return OrderedDict([("0",rP7),("1",rP6),("2",rP5),("3",rP4),("4",rP3)])


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
    testInput = torch.randn(size=[5,3,32,32]).float()
    model = EfficientNetReform(in_channels=3,w=2,d=2)
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

