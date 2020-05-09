import torch
import torch.nn as nn
import torch.nn.functional as F
from Tools import Conv2dDynamicSamePadding
from collections import OrderedDict
from BiFPN import BiFPN
from Tools import Blur_Pooling
from MobileNetV2 import Inverted_Residual_Block
from ResNest import ResNest_Block


class MBConvBlock(nn.Module):

    def __init__(self, in_channels,dropRate = 0.2,expansion_factor = 2):
        super().__init__()
        ### M
        self.InRB = Inverted_Residual_Block(in_channels,in_channels,dropRate,expansion_factor)
        ### R
        self.resNeSt = ResNest_Block(k=4, r=4, in_channels=in_channels,drop_p=dropRate)


    def forward(self, x):
        xInRB = self.InRB(x)
        xResnest = self.resNeSt(xInRB)
        return xResnest

class MB_Layers(nn.Module):

    def __init__(self,in_channels,layers,drop_connect_rate = 0.2):
        super().__init__()
        blocksDic = OrderedDict()
        for i in range(layers):
            blocksDic[str(i)] = MBConvBlock(in_channels, drop_connect_rate)
        self.MBs = nn.Sequential(blocksDic)

    def forward(self, x):
        return self.MBs(x)

class Blur_Down_Sample(nn.Module):

    def __init__(self,in_channels,out_channels,pooling_type):
        super().__init__()
        self.downSample = nn.Sequential(Conv2dDynamicSamePadding(in_channels,out_channels,kernel_size=3,bias=False),
                                   nn.BatchNorm2d(out_channels,eps=1e-3,momentum=1e-2),
                                   Blur_Pooling(out_channels,pooling_type=pooling_type))

    def forward(self,x):
        return self.downSample(x)


class EfficientNetReform(nn.Module):

    def __init__(self,in_channels,w = 3,d = 3,drop_connect_rate = 0.2,num_classes = 10,classify = True):
        super(EfficientNetReform,self).__init__()
        ### r0 32
        self.conv_stem = Conv2dDynamicSamePadding(in_channels, 32 * w, kernel_size=7, stride=1)
        self.bn0 = nn.BatchNorm2d(32 * w,  eps=0.001,momentum=0.01)
        self.trans1 = Blur_Down_Sample(32 * w, 64 * w,pooling_type="Max")
        ### blocks
        ### r1 16
        self.block1 = MB_Layers(64 * w, layers=2 * d,  drop_connect_rate=drop_connect_rate)
        self.block2 = MB_Layers(64 * w, layers=3 * d,  drop_connect_rate=drop_connect_rate)
        self.trans2 = Blur_Down_Sample(64 * w, 80 * w,pooling_type="Avg")
        ### r2 8
        self.block3 = MB_Layers(80 * w, layers=3 * d,  drop_connect_rate=drop_connect_rate)
        self.block4 = MB_Layers(80 * w, layers=4 * d,  drop_connect_rate=drop_connect_rate)
        self.trans3 = Blur_Down_Sample(80 * w, 192 * w, pooling_type="Avg")
        ### r3 4
        self.block5 = MB_Layers(192 * w, layers=4 * d, drop_connect_rate=drop_connect_rate)
        self.block6 = MB_Layers(192 * w, layers=5 * d, drop_connect_rate=drop_connect_rate)
        self.trans4 = Blur_Down_Sample(192 * w, 320 * w, pooling_type="Avg")
        ### r4 2
        self.block7 = MB_Layers(320 * w, layers=3 * d,  drop_connect_rate=drop_connect_rate)
        self.block8 = MB_Layers(320 * w, layers=2 * d, drop_connect_rate=drop_connect_rate)
        ### BiFPN
        self.BifpnFirst = BiFPN(num_channels=560 + 64 * w, conv_channels=[80 * w, 192 * w, 320 * w],first_time=True)
        self.Bifpn = BiFPN(560 + 64 * w, conv_channels=[], first_time=False)
        ### classify
        self.classify = classify
        if classify :
            self.resP3 = nn.Sequential(MBConvBlock(560 + 64 * w,drop_connect_rate),
                                       Blur_Down_Sample(560 + 64 * w,560 + 64 * w,pooling_type="AVG"),
                                       MBConvBlock(560 + 64 * w, drop_connect_rate),
                                       Blur_Down_Sample(560 + 64 * w, 560 + 64 * w, pooling_type="MAX"))
            self.resP4 = nn.Sequential(MBConvBlock(560 + 64 * w,drop_connect_rate),
                                       Blur_Down_Sample(560 + 64 * w,560 + 64 * w,pooling_type="MAX"))
            self.seq = nn.Sequential(nn.Linear(560 + 64 * w,1280),
                                     nn.BatchNorm1d(1280,eps=0.001,momentum=0.01),
                                     nn.Dropout(drop_connect_rate),
                                     nn.Linear(1280,num_classes))

    def forward(self,x):
        """
        :param x:
        :return:
        """
        #print(x.shape)
        p1= self.bn0(self.conv_stem(x))
        trans1 = self.trans1(p1)
        p2 = self.block2(self.block1(trans1))
        trans2 = self.trans2(p2)
        p3 = self.block4(self.block3(trans2))
        trans3 = self.trans3(p3)
        p4 = self.block6(self.block5(trans3))
        trans4 = self.trans4(p4)
        p5 = self.block8(self.block7(trans4))
        # print(p3.shape)
        # print(p4.shape)
        # print(p5.shape)
        p3_f, p4_f, p5_f = self.BifpnFirst(p3 = p3,p4 = p4,p5 = p5)
        # print(p3_f.shape)
        # print(p4_f.shape)
        # print(p5_f.shape)
        p3_s, p4_s, p5_s = self.Bifpn(p3 = p3_f, p4 = p4_f, p5 =  p5_f)
        # print(p3_s.shape)
        # print(p4_s.shape)
        # print(fea5.shape)
        if self.classify:
            fea3 = self.resP3(p3_s.clone())
            fea4 = self.resP4(p4_s.clone())
            feat = fea3 + fea4 + p5_s
            return self.seq(F.adaptive_avg_pool2d(feat,[1,1]).squeeze(-1).squeeze(-1).contiguous())
        else:
            return OrderedDict([("0",p5_s),("1",p4_s),("2",p3_s)])




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
    print("Min grad : ",min(ave_grads))
    print("Max ",max(ave_grads))
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
    model = EfficientNetReform(in_channels=3,w=2,d=3)
    optimizer = rmsprop.RMSprop(model.parameters(), 5e-4, momentum=0.9, weight_decay=1e-5)
    outputs = model(testInput)
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
    plot_grad_flow(model.named_parameters())

