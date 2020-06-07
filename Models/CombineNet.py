import torch
import torch.nn as nn
from Models.MobileNetV2 import MobileNetV2
from Models.ResNest import ResNestNet

class CombineNet(nn.Module):

    def __init__(self,in_channels,num_classes,dropRate,w,d):
        super().__init__()
        self.mobileNet = MobileNetV2(in_channels,num_classes,dropRate,w * 3,d)
        self.resnestNet = ResNestNet(in_channels,num_classes,dropRate,w,d)


    def forward(self,x):
        xRes = x.clone()
        resnestR = self.resnestNet(xRes)
        mobileR = self.mobileNet(x)
        return (resnestR  + mobileR ) / 2.

class CombineNet_MultiGPUs(nn.Module):

    def __init__(self,in_channels,num_classes,dropRate,w,d,device0= "cuda:0",device1 = "cuda:1"):
        super().__init__()
        self.device0 = device0
        self.device1 = device1
        self.mobileNet = MobileNetV2(in_channels,num_classes,dropRate,w * 3,d).to(device0)
        self.resnestNet = ResNestNet(in_channels,num_classes,dropRate,w,d).to(device1)
        self.gate = nn.Parameter(torch.ones(size=[2]).float(),requires_grad=True).to(device0)

    def forward(self,x):
        resnestR = self.resnestNet(x)
        xMB = x.clone().to(self.device0)
        mobileR = self.mobileNet(xMB)
        gate = torch.sigmoid(self.gate)
        return resnestR.to(self.device0) * gate[0] + mobileR * gate[1]


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
    testInput = torch.randn(size=[5,3,32,32]).float()
    model = CombineNet(3,10,0.5,1,2)
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

