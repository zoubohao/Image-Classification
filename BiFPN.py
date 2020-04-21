import torch
import torch.nn as nn
from Tools import MatchingAuto
from Tools import SeparableConvBlock
from Tools import Conv2dStaticSamePadding
from Tools import Swish


class BiFPN(nn.Module):
    """
    modified by Bohao Zou
    """

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4):
        """
        Args:
            num_channels: in channels in BiFPN
            conv_channels: input cov channels.
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.matching = MatchingAuto()
        # Conv layers
        self.conv4_up = SeparableConvBlock(num_channels)
        self.conv3_up = SeparableConvBlock(num_channels)
        self.conv4_down = SeparableConvBlock(num_channels)
        self.conv5_down = SeparableConvBlock(num_channels)

        # Feature scaling layers

        self.p4_downsample = nn.Sequential(Conv2dStaticSamePadding(num_channels,num_channels,3,2,groups=num_channels,bias=False),
                                           nn.Conv2d(num_channels,num_channels,1,1,0,bias=True),
                                           nn.BatchNorm2d(num_channels,eps=1e-3,momentum=0.01),
                                           Swish())
        self.p5_downsample = nn.Sequential(Conv2dStaticSamePadding(num_channels,num_channels,3,2,groups=num_channels,bias=False),
                                           nn.Conv2d(num_channels,num_channels,1,1,0,bias=True),
                                           nn.BatchNorm2d(num_channels,eps=1e-3,momentum=0.01),
                                           Swish())


        self.swish1 =  Swish()
        self.swish2 = Swish()
        self.swish3 = Swish()
        self.swish4 = Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels,eps=1e-3,momentum=0.01),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels,eps=1e-3,momentum=0.01),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels,eps=1e-3,momentum=0.01),
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels,eps=1e-3,momentum=0.01),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels,eps=1e-3,momentum=0.01),
            )

        # Weight
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()


    def forward(self, p3,p4,p5):
        """
        illustration of a minimal bifpn unit
            P5_0 -------------------------->P5_2 -------->
               |-------------|                ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """


        if self.first_time:
            p3, p4, p5 = p3 , p4, p5

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0
            p3_in, p4_in, p5_in= p3 , p4, p5


        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.swish1(weight[0] * p4_in + weight[1] * self.matching(p5_in, p4_in.shape[-2:])))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish2(weight[0] * p3_in + weight[1] * self.matching(p4_up, p3_in.shape[-2:])))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish3(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.matching(self.p4_downsample(p3_out),p4_in.shape[-2:])))

        # Weights for P5_0, P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish4(weight[0] * p5_in + weight[1] * self.matching(self.p5_downsample(p4_out),p5_in.shape[-2:])))

        return p3_out, p4_out, p5_out



if __name__ == "__main__":
    testP3 = torch.ones(size=[3, 16, 8, 8])
    testP4 = torch.ones(size=[3, 32, 4, 4])
    testP5 = torch.ones(size=[3, 64, 2, 2])
    testBiFPN = BiFPN(num_channels=128,conv_channels=[16, 32,64],first_time=True)
    rP3, rP4, rP5 = testBiFPN(testP3, testP4, testP5)
    rP3, rP4, rP5 = BiFPN(num_channels=128,conv_channels=[],first_time=False)(rP3, rP4, rP5)
    print(rP3.shape)
    print(rP4.shape)
    print(rP5.shape)
