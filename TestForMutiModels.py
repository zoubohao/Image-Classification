from MobileNetV2 import MobileNetV2
from ResNest import ResNestNet
import torchvision as tv
from torch.utils.data import DataLoader
import torch


labelsNumber = 10
modelRes = ResNestNet(3,labelsNumber,0.47,1,2)
modelMB = MobileNetV2(3,labelsNumber,0.45,2,2)
modelRes.load_state_dict(torch.load("./Model_Weight/Model_Res0.9051.pth"))
modelMB.load_state_dict(torch.load("./Model_Weight/Model_MB0.9258.pth"))
transformationTest = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
cifarDataTestSet = tv.datasets.CIFAR10(root="./Cifar10/",train=False,download=True,transform=transformationTest)
testloader = DataLoader(cifarDataTestSet, batch_size=50, shuffle=False)

total = 0
correct = 0
modelRes = modelRes.eval()
modelMB = modelMB.eval()
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputsRes, targets = inputs, targets
    inputsMB = inputs
    outputsMB = modelMB(inputsMB)
    outputsRes = modelRes(inputsRes)
    finalOut = outputsMB * 0.51 + outputsRes * 0.49
    _, predicted = finalOut.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    print('Acc: %.3f %% (%d / %d)'% (100. * correct / total, correct, total))
acc = correct / total


