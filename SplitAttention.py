import torch
import torch.nn as nn
import torch.nn.functional as F


def Swish(x):
    return x * torch.sigmoid(x)

class SplitAttention(nn.Module):

    def __init__(self,K,in_channels,inner_channels):
        super().__init__()
        self.K = K
        self.splitConv = nn.Conv2d(in_channels,in_channels * K,3,1,1,groups=K)
        self.gn0 = nn.GroupNorm(K,in_channels * K,eps=1e-3)
        self.fc1 = nn.Conv2d(in_channels, inner_channels,1)
        self.gn1 = nn.GroupNorm(K,inner_channels,eps=1e-3)
        self.fc2 = nn.Conv2d(inner_channels, in_channels * K, 1, groups=self.K)
        self.in_channels = in_channels
        self.inner_channels = inner_channels

    def forward(self,x):
        x = Swish(self.gn0(self.splitConv(x)))
        split1 = torch.split(x,dim=-3,split_size_or_sections=self.in_channels)
        sumTensor = sum(split1)
        avgPool = F.adaptive_avg_pool2d(sumTensor,[1,1])
        fc1 = Swish(self.gn1(self.fc1(avgPool)))
        attenAll = self.fc2(fc1)
        split2 = torch.split(attenAll,split_size_or_sections=self.in_channels,dim=-3)
        attenedList = []
        for i,one in enumerate(split2):
            thisAtten = F.softmax(one.view(-1,self.in_channels),dim=-1)
            attenedList.append(thisAtten.view(-1,self.in_channels,1,1) * split1[i])
        return sum(attenedList)

if __name__ == "__main__":
    testInput = torch.randn(size=[5,16,31,31]).float()
    testModule = SplitAttention(K=4,in_channels=16,inner_channels=32)
    print(testModule(testInput).shape)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    writer.add_graph(testModule, testInput)
    writer.close()









