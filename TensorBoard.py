import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from EfficientNet import EfficientNetReform
from BiFPN import BiFPN

writer = SummaryWriter()
model = EfficientNetReform(3,2,2)
testInput = torch.randn(size=[8,3,32,32]).float()
writer.add_graph(model,testInput)
writer.close()

# writer = SummaryWriter()
# testP3 = torch.ones(size=[3, 16, 129, 129]).float()
# testP4 = torch.ones(size=[3, 32, 69, 69]).float()
# testP5 = torch.ones(size=[3, 64, 24, 24]).float()
# testBiFPN = BiFPN(num_channels=128, conv_channels=[16, 32, 64], first_time=True)
# writer.add_graph(model=testBiFPN,input_to_model=(testP3, testP4, testP5))
# writer.close()


