import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from EfficientNet import EfficientNetReform

writer = SummaryWriter()
model = EfficientNetReform(3,1)
testInput = torch.randn(size=[8,3,32,32]).float()
writer.add_graph(model,testInput)
writer.close()

