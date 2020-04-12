import torch
import torchvision as tv
from torch.utils import data as d
import numpy as np
import Model
import torch.nn as nn
import torch.optim.rmsprop as rmsprop
from sklearn import metrics
import EfficientNet
from prefetch_generator import BackgroundGenerator
from MyLoss import SimulateCenterLoss

class DataLoaderX(d.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

if __name__ == "__main__":
    ### config
    batchSize = 8
    tMaxIni = 1200
    growthRate = 32
    blocks = [6,12,24,16]
    learning_rate = 1e-4
    labelsNumber = 10
    ifUseBn = True
    ifTrain = True
    epoch = 50
    displayTimes = 25
    modelSavePath = "./BN/"
    loadWeight = True
    trainModelLoad = 3
    testModelLoad = 10
    decayRate = 0.93
    fy = 5
    stepTimes =2
    ### Data pre-processing
    transformationTrain = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(p = 0.25),
        tv.transforms.RandomVerticalFlip(p = 0.334),
        #tv.transforms.RandomApply([tv.transforms.RandomResizedCrop(32)],p=0.5),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([.5,.5,.5],[.5,.5,.5])
    ])
    transformationTest = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ])
    #cifarDataTrainSet = tv.datasets.CIFAR100(root="./Cifar100/",train=True,download=True,transform=transformation)
    #cifarDataTestSet = tv.datasets.CIFAR100(root="./Cifar100/",train=False,download=True)
    cifarDataTrainSet = tv.datasets.CIFAR10(root="./Cifar10/",train=True,download=True,transform=transformationTrain)
    cifarDataTestSet = tv.datasets.CIFAR10(root="./Cifar10/",train=False,download=True,transform=transformationTest)
    dataLoader = DataLoaderX(cifarDataTrainSet,batch_size=batchSize,shuffle=True,num_workers=4,pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### model construct
    model = Model.MyModel(3,labelsNumber,blocks,growthRate,ifUseBn).to(device)
    #model = EfficientNet.EfficientNet(3,labelsNumber,fy=fy,dropOutRate=0.2).to(device)
    print(model)
    #lossCri = Model.LabelsSmoothingCrossLoss(labelsNumber,0.09).to(device)
    #lossCri = nn.CrossEntropyLoss(reduction="sum")
    lossCri = SimulateCenterLoss(labelsNumber,1024,0.4).to(device)
    optimizer = rmsprop.RMSprop(model.parameters(),learning_rate,momentum=0.9,weight_decay=1e-5)
    if loadWeight :
       # print(torch.load(modelSavePath + "Model_BN_" + str(trainModelLoad) + ".pth"))
        model.load_state_dict(torch.load(modelSavePath + "Model_" + str(trainModelLoad) + ".pth"))
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0.)
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0.)
    ### Train or Test
    scheduler = Model.CosineDecaySchedule(5e-6,learning_rate,tMaxIni,1.2,lrDecayRate=decayRate)
    if ifTrain:
        model.train()
        trainingTimes = 0
        for e in range(epoch):
            for times , (images, labels) in enumerate(dataLoader):
                #optimizer.zero_grad()
                imagesCuda = images.float().to(device,non_blocking = True)
                labelsCuda = labels.long().to(device,non_blocking = True)
                #scheduler.step(e + times // iters)
                predict , features = model(imagesCuda)
                oriLoss = lossCri(predict,features,labelsCuda)
                loss = oriLoss / stepTimes
                loss.backward()
                #optimizer.step()
                if trainingTimes % displayTimes == 0:
                    print("#########")
                    print("Predict is : ",predict[0:3])
                    print("Labels are : ",labelsCuda[0:3])
                    print("Learning rate is ", optimizer.state_dict()['param_groups'][0]["lr"])
                    print("Loss is ", oriLoss)
                    print("Epoch : ", e)
                    print("Training time is ", trainingTimes)
                trainingTimes += 1
                if trainingTimes % stepTimes == 0:
                    learning_rate = scheduler.calculateLearningRate()
                    state_dic = optimizer.state_dict()
                    state_dic["param_groups"][0]["lr"] = learning_rate
                    optimizer.load_state_dict(state_dic)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            torch.save(model.state_dict(), modelSavePath + "Model_" + str(e) + ".pth")
    else:
        model.eval()
        model.load_state_dict(torch.load(modelSavePath + "Model_" + str(testModelLoad) + ".pth"))
        predictList = []
        truthList = []
        k = 0
        for testImage, testTarget in cifarDataTestSet:
            predictTensor = model(testImage.unsqueeze(0).to(device)).cpu().detach().numpy()
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









