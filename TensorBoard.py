import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from EfficientNet import EfficientNet

writer = SummaryWriter()
model = EfficientNet(3,10,fy=1)
testInput = torch.randn(size=[8,3,32,32]).float()
writer.add_graph(model,testInput)
writer.close()

