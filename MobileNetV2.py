import torch.nn as nn
from Tools import Conv2dDynamicSamePadding
from Tools import Mish
import torch.nn.functional as F
import math
from Tools import drop_connect_B


class Inverted_Residual_Block(nn.Module):

    def __init__(self,in_channels,out_channels,stride = 1,expansion_factor = 2,drop_connect_rate = 0.5):
        super().__init__()
        self.expension = nn.Sequential(Conv2dDynamicSamePadding(in_channels,in_channels * expansion_factor,
                                                               kernel_size=1,stride=1,bias=False),
                                       nn.BatchNorm2d(in_channels * expansion_factor,eps=1e-3,momentum=1e-2),
                                       Mish())
        self.depthwise_conv = nn.Sequential(Conv2dDynamicSamePadding(in_channels* expansion_factor, in_channels* expansion_factor,
                                                      kernel_size=3, stride=stride, groups=in_channels * expansion_factor, bias=False),
                                            nn.GroupNorm(num_groups=in_channels * expansion_factor,
                                                         num_channels=in_channels* expansion_factor,eps=1e-3,affine=True),
                                            Mish())
        self.pointwise_conv = nn.Sequential(Conv2dDynamicSamePadding(in_channels * expansion_factor, out_channels, kernel_size=1, stride=1),
                                            nn.BatchNorm2d(out_channels,eps=1e-3,momentum=1e-2))
        self.drop_conn = drop_connect_rate
        self.shortCut = True
        if in_channels != out_channels :
            self.shortCut = False

    def forward(self,x):
        if self.shortCut:
            out = self.pointwise_conv(self.depthwise_conv(self.expension(x)))
            if self.training and self.drop_conn > 0:
                out = drop_connect_B(out,self.drop_conn)
            return out + x.clone()
        else:
            return  self.pointwise_conv(self.depthwise_conv(self.expension(x)))


class Bottleneck(nn.Module):

    def __init__(self,in_channels,out_channels,layers,stride = 1,t = 2,drop_connect_rate = 0.5):
        super().__init__()
        blocks = [Inverted_Residual_Block(in_channels,out_channels,stride,t,drop_connect_rate)]
        for _ in range(layers-1):
            blocks.append(Inverted_Residual_Block(out_channels,out_channels,1,t,drop_connect_rate))
        self.bottleNeck = nn.Sequential(*blocks)


    def forward(self,x):
        return self.bottleNeck(x)

class MobileNetV2(nn.Module):

    def __init__(self,in_channels,num_classes,dropRate,w = 1,d = 1):
        super().__init__()
        self.conv1 = nn.Sequential(Conv2dDynamicSamePadding(in_channels,32 * w,7,1),
                                   nn.BatchNorm2d(32 * w,eps=1e-3,momentum=1e-2),
                                   Mish())
        self.b1 = Bottleneck(32 * w,16 * w,1 * d,1,1,dropRate)
        self.b2 = Bottleneck(16 * w,24 * w,2 * d,2,6,dropRate)
        self.b3 = Bottleneck(24 * w,32 * w,3 * d,2,6,dropRate)
        self.b4 = Bottleneck(32 * w,64 * w,4 * d,2,6,dropRate)
        self.b5 = Bottleneck(64 * w,96 * w,3 * d,1,6,dropRate)
        self.b6 = Bottleneck(96 * w,160 * w,3 * d,2,6,dropRate)
        self.b7 = Bottleneck(160 * w,320 * w,1 * d,1,6,dropRate)
        self.conv2 = nn.Sequential(nn.Conv2d(320 * w,1280,1,1,bias=False),
                                   nn.BatchNorm2d(1280),
                                   Mish())
        self.dropout = nn.Dropout(dropRate + 0.3,True)
        self.fc = nn.Linear(1280,num_classes)
        self._initialize_weights()

    def forward(self,x):
        conv1 = self.conv1(x)
        #print(conv1.shape)
        b1 = self.b1(conv1)
        b2 = self.b2(b1)
        #print(b2.shape)
        b3 = self.b3(b2)
        #print(b3.shape)
        b4 = self.b4(b3)
        b5 = self.b5(b4)
        #print(b5.shape)
        b6 = self.b6(b5)
        b7 = self.b7(b6)
        #print(b7.shape)
        conv2 = self.conv2(b7)
        avgG = F.adaptive_avg_pool2d(conv2,[1,1])
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
    import torch
    testInput = torch.ones(size=[5,3,32,32]).float()
    testModule = MobileNetV2(3,10,0.5,w=3,d=2)
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




