import torch.nn as nn
from Tools import Conv2dDynamicSamePadding


class Inverted_Residual_Block(nn.Module):

    def __init__(self,in_channels,out_channels,dropRate = 0.2,expansion_factor = 2):
        super().__init__()
        self.expension = nn.Sequential(Conv2dDynamicSamePadding(in_channels,in_channels * expansion_factor,
                                                               kernel_size=1,stride=1,bias=False),
                                       nn.BatchNorm2d(in_channels * expansion_factor,eps=0.001,momentum=0.01),
                                       nn.ReLU6(inplace=True))
        self.depthwise_conv = nn.Sequential(Conv2dDynamicSamePadding(in_channels* expansion_factor, in_channels* expansion_factor,
                                                      kernel_size=3, stride=1, groups=in_channels* expansion_factor, bias=False),
                                            nn.BatchNorm2d(in_channels * expansion_factor,eps=0.001,momentum=0.01),
                                            nn.ReLU6(inplace=True))
        self.pointwise_conv = nn.Sequential(Conv2dDynamicSamePadding(in_channels * expansion_factor, out_channels, kernel_size=1, stride=1),
                                            nn.BatchNorm2d(out_channels,eps=0.001,momentum=0.01))
        self.dropout = nn.Dropout2d(dropRate)

    def forward(self,x):
        xOri = x.clone()
        return  self.dropout(xOri + self.pointwise_conv(self.depthwise_conv(self.expension(x))))






