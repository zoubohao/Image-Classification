import torch
import torch.nn as nn
from Tools import SeparableConvBlock
from Tools import Conv2dDynamicSamePadding
from Tools import Pool2dStaticSamePadding
from Tools import Swish

class BiFPN(nn.Module):
    """
    modified by Bohao Zou
    """

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # Conv layers
        self.conv4_up = SeparableConvBlock(num_channels)
        self.conv3_up = SeparableConvBlock(num_channels)
        self.conv4_down = SeparableConvBlock(num_channels)
        self.conv5_down = SeparableConvBlock(num_channels)

        # Feature scaling layers
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = Pool2dStaticSamePadding(3, 2,"MAX")
        self.p5_downsample = Pool2dStaticSamePadding(3, 2,"MAX")

        self.swish1 = Swish()
        self.swish2 = Swish()
        self.swish3 = Swish()
        self.swish4 = Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dDynamicSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dDynamicSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dDynamicSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels),
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


    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P5_0 -----------------*--------> P5_2 -------->
               |--------*----|                ↑
                             ↓                *
            P4_0 -----*----> P4_1 ----*----> P4_2 -------->
               |-------------|-------*------↑ ↑
                             |------*-------↓ *
            P3_0 -------------*-----------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation
        if self.first_time:
            p3,p4,p5 = inputs
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            p3, p4, p5 = inputs
            # P3_0, P4_0, P5_0
            p3_in = p3.clone()
            p4_in = p4.clone()
            p5_in = p5.clone()

        # Weights for P4_0 and P5_0 to P4_1/
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.swish1(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_in)))

        # Weights for P3_0 and P4_1 to P3_2/
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish2(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))


        # Weights for P4_0, P4_1 and P3_2 to P4_2/
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish3(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0 and P4_2 to P5_2/
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish4(weight[0] * p5_in  + weight[1] * self.p5_downsample(p4_out)))

        return p3_out, p4_out, p5_out



if __name__ == "__main__":
    testP3 = torch.ones(size=[5,16,8,8])
    testP4 = torch.ones(size=[5, 32, 4, 4])
    testP5 = torch.ones(size=[5, 64, 2, 2])
    testBF = BiFPN(128,[16,32,64],first_time=True)
    testB = BiFPN(128,[],first_time=False)
    p3f,p4f,p5f = testBF(testP3,testP4,testP5)
    print(p3f.shape)
    print(p4f.shape)
    print(p5f.shape)
    p3s,p4s,p5s = testB(p3f,p4f,p5f)
    print(p3s.shape)
    print(p4s.shape)
    print(p5s.shape)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    writer.add_graph(testBF, (testP3,testP4,testP5))
    writer.close()

