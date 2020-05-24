import torch
import torchvision as tv
import numpy as np
from CosineSchedule import CosineDecaySchedule
import torch.nn as nn
from sklearn import metrics
import torch_optimizer as optim
from torch.utils.data import DataLoader
from ResNest import ResNestNet



# 0.01 for epoch [0,150)
# # 0.001 for epoch [150,250)
# # 0.0001 for epoch [250,350)
if __name__ == "__main__":
    ### config
    w = 1
    d = 2
    batchSize = 92
    labelsNumber = 10
    epoch = 150
    displayTimes = 10
    drop_connect_rate = 0.25
    reg_lambda = 5e-4
    split = 43
    reduction = 'mean'
    if_sgd = False
    ###
    modelSavePath = "./Model_Weight/"
    saveTimes = 2000 # For training 2000 step
    ###
    loadWeight= False
    trainModelLoad = 0.6965
    ### trainingTimes = stepTimes * currentStep
    tMaxIni = 1000
    maxLR = 1e-2
    minLR = 1e-2
    decayRate = 0.96
    ## warmUpSteps * stepTimes = trainingTimes
    warmUpSteps = 1000
    ###
    stepTimes = 1
    ###
    ifTrain = True
    testModelLoad = 3
    device1 = "cuda:1"


    ### Data pre-processing
    transformationTrain = tv.transforms.Compose([
    tv.transforms.RandomCrop(32, padding=4),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    transformationTest = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
    ])


    cifarDataTrainSet = tv.datasets.CIFAR10(root="./Cifar10/",train=True,download=True,transform=transformationTrain)
    cifarDataTestSet = tv.datasets.CIFAR10(root="./Cifar10/",train=False,download=True,transform=transformationTest)
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataLoader = DataLoader(cifarDataTrainSet,batch_size=batchSize,shuffle=True,num_workers=6,pin_memory=True)
    testloader = DataLoader(cifarDataTestSet, batch_size=100, shuffle=False, num_workers=2)

    model = ResNestNet(3,labelsNumber,drop_connect_rate,w,d).to(device1)


    lossCri = nn.CrossEntropyLoss(reduction=reduction).to(device1)
    if if_sgd:
        optimizer = torch.optim.SGD(model.parameters(),lr = minLR,momentum=0.9,nesterov=True,weight_decay=reg_lambda)
    else:
        optimizer = optim.adabound.AdaBound(model.parameters(),lr=minLR,final_lr=minLR,weight_decay=reg_lambda)
    if loadWeight :
        model.load_state_dict(torch.load(modelSavePath + "Model_Res" + str(trainModelLoad) + ".pth"))


    ### Train or Test
    scheduler = CosineDecaySchedule(lrMin=minLR,lrMax=maxLR,tMaxIni=tMaxIni,factor=1.15,lrDecayRate=decayRate,warmUpSteps=warmUpSteps)
    if ifTrain:
        model = model.train(mode=True)
        trainingTimes = 0
        optimizer.zero_grad()
        for e in range(epoch):
            for times , (images, labels) in enumerate(dataLoader):
                imagesCuda = images.float().to(device1,non_blocking = True)
                labelsCuda = labels.long().to(device1,non_blocking = True)
                predict = model(imagesCuda)
                criLoss = lossCri(predict,labelsCuda)
                tloss = torch.div(criLoss,stepTimes)
                tloss.backward()
                if trainingTimes % displayTimes == 0:
                    with torch.no_grad():
                        _, predicted = predict.max(1)
                        total = labelsCuda.size(0)
                        correct = predicted.eq(labelsCuda).sum().item()
                        print("######################")
                        print("Epoch : %d , Training time : %d" % (e, trainingTimes))
                        print("Cri Loss is %.3f " % (criLoss.item()))
                        print("Learning rate is ", optimizer.state_dict()['param_groups'][0]["lr"])
                        print("Correct ratio is %3f "% (correct / total + 0.))
                        print("predicted labels : ",predict[0:2,])
                        print("Truth labels : ",labelsCuda[0:2,])
                trainingTimes += 1
                if trainingTimes % stepTimes == 0:
                    learning_rate = scheduler.calculateLearningRate()
                    state_dic = optimizer.state_dict()
                    state_dic["param_groups"][0]["lr"] = learning_rate
                    optimizer.load_state_dict(state_dic)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                if trainingTimes % saveTimes == 0 :
                    ### val part
                    model = model.eval()
                    test_loss = 0
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(testloader):
                            inputs, targets = inputs.to(device1), targets.to(device1)
                            outputs = model(inputs)
                            loss = lossCri(outputs, targets)
                            test_loss += loss.item()
                            _, predicted = outputs.max(1)
                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()
                            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                    acc =  correct / total
                    torch.save(model.state_dict(), modelSavePath + "Model_Res" + str(acc) + ".pth")
                    model = model.train(mode=True)
        torch.save(model.state_dict(), modelSavePath + "Model_ResF" + ".pth")
    else:
        model = model.eval()
        model.load_state_dict(torch.load(modelSavePath + "Model_Res" + str(testModelLoad) + ".pth"))
        predictList = []
        truthList = []
        k = 0
        for testImage, testTarget in cifarDataTestSet:
            predictTensor = model(testImage.unsqueeze(0).to(device1)).cpu().detach().numpy()
            position = np.argmax(np.squeeze(predictTensor))
            if k % displayTimes == 0:
                print("##############" + str(k))
                print(position)
                print(testTarget)
            predictList.append(position)
            truthList.append(testTarget)
            k += 1
        acc = metrics.accuracy_score(y_true=truthList,y_pred=predictList)
        classifiedInfor = metrics.classification_report(y_true=truthList,y_pred=predictList)
        macroF1 = metrics.f1_score(y_true=truthList,y_pred=predictList,average="macro")
        microF1 = metrics.f1_score(y_true=truthList,y_pred=predictList,average="micro")
        print("The accuracy is : ",acc)
        print("The classified result is : ")
        print(classifiedInfor)
        print("The macro F1 is : ",macroF1)
        print("The micro F1 is : ",microF1)





