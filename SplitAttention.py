import torch
import torch.nn as nn
import torch.nn.functional as F
from Tools import Swish
from Tools import AddN

def sumMax(x,dim = -1):
    reluX = F.relu(x)
    sumT = torch.sum(reluX,dim=dim,keepdim=True) + 1e-4
    return reluX / sumT


class SplitAttention(nn.Module):

    def __init__(self,r,in_channels,inner_channels):
        super().__init__()
        self.r = r
        self.fc1 = nn.Conv2d(in_channels, inner_channels,1)
        self.gn1 = nn.BatchNorm2d(inner_channels,eps=1e-3,momentum=0.01)
        self.fc2 = nn.Conv2d(inner_channels, in_channels * r, 1, groups=r)
        self.in_channels = in_channels
        self.act = Swish()

    def forward(self,inputs):
        sumTensor = AddN(inputs)
        avgPool = F.adaptive_avg_pool2d(sumTensor,[1,1])
        fc1 = self.act(self.gn1(self.fc1(avgPool)))
        attenAll = self.fc2(fc1)
        split2 = torch.chunk(attenAll,chunks=self.r,dim=-3)
        attenedList = []
        for i,one in enumerate(split2):
            thisAtten = sumMax(one.view(-1,self.in_channels),dim=-1)
            attenedList.append(thisAtten.view(-1,self.in_channels,1,1) * inputs[i])
        return AddN(attenedList)


class Cardinal(nn.Module):

    def __init__(self,R,in_channels):
        super().__init__()
        self.r = R
        self.seq1 = nn.Sequential(nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=R),
                                   nn.BatchNorm2d(in_channels,0.001,0.01))
        self.seq3 = nn.Sequential(nn.Conv2d(in_channels,in_channels * R,kernel_size=3,stride=1,groups=R,padding=1),
                                   nn.BatchNorm2d(in_channels * R,0.001,0.01),
                                   Swish())
        self.splitAttention = SplitAttention(R,in_channels,inner_channels=in_channels * 2)

    def forward(self, x):
        return self.splitAttention(torch.chunk(self.seq3(self.seq1(x)),chunks=self.r,dim=-3))


class ResNeSt(nn.Module):

    def __init__(self,k,r,in_channels):
        super().__init__()
        self.k = k
        self.Cardinals = nn.ModuleList([Cardinal(r,in_channels // k) for _ in range(k)])
        self.seq = nn.Sequential(nn.Conv2d(in_channels,in_channels,1,1,groups=k * r),
                                 nn.BatchNorm2d(in_channels,0.001,0.01))

    def forward(self, x):
        xOri = x.clone()
        chunks = torch.chunk(x,self.k,dim=-3)
        catList = []
        for i , one in enumerate(chunks):
            catList.append(self.Cardinals[i](one))
        return self.seq(torch.cat(catList,dim=-3)) + xOri


if __name__ == "__main__":
    testInput = torch.randn(size=[5,16,31,31]).float()
    testModule = ResNeSt(k = 4,r= 4,in_channels=16)
    testRes = testModule(testInput)
    testRes.mean().backward()
    from Tools import plot_grad_flow
    plot_grad_flow(testModule.named_parameters())
    print(testModule(testInput).shape)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    writer.add_graph(testModule, testInput)
    writer.close()
    testTensor = torch.rand(size=[3,6]).float()
    print(sumMax(testTensor))









