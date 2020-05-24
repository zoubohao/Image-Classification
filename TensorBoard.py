import torch
from torch.utils.tensorboard import SummaryWriter
from CombineNet import CombineNet
from BiFPN import BiFPN
from ResNest import ResNestNet
from ResNestOri import resnest101

writer = SummaryWriter()
model =  ResNestNet(3,10,0.5,1,2)
model = model.train(True)
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


