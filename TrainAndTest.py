import torch
import torchvision as tv
from torch.utils import data as d
import numpy as np
from CosineSchedule import CosineDecaySchedule
import torch.nn as nn
import torch.optim.rmsprop as rmsprop
from sklearn import metrics
import EfficientReformModel
from prefetch_generator import BackgroundGenerator


class DataLoaderX(d.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

if __name__ == "__main__":
    ### config
    w = 2
    d = 3
    batchSize = 32
    labelsNumber = 10
    epoch = 50
    displayTimes = 10
    ###
    modelSavePath = "./Model_Weight/"
    saveTimes = 1200
    ###
    loadWeight = False
    trainModelLoad = 0.096
    ###
    valTestSampleNumber = 500
    ###
    tMaxIni = 1100
    maxLR = 1e-3
    minLR = 1e-6
    decayRate = 0.9
    warmUpSteps = 1200
    ###
    stepTimes = 1
    ###
    ifTrain = True
    testModelLoad = 3


    ### Data pre-processing
    transformationTrain = tv.transforms.Compose([
        #tv.transforms.Resize(size=[32 * 2,32 * 2]),
        tv.transforms.RandomHorizontalFlip(p = 0.25),
        tv.transforms.RandomVerticalFlip(p = 0.334),
        tv.transforms.RandomApply([tv.transforms.RandomResizedCrop(32)],p=0.5),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    transformationTest = tv.transforms.Compose([
        #tv.transforms.Resize(size=[32 * 2,32 * 2]),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    #cifarDataTrainSet = tv.datasets.CIFAR100(root="./Cifar100/",train=True,download=True,transform=transformation)
    #cifarDataTestSet = tv.datasets.CIFAR100(root="./Cifar100/",train=False,download=True)

    cifarDataTrainSet = tv.datasets.CIFAR10(root="./Cifar10/",train=True,download=True,transform=transformationTrain)
    cifarDataTestSet = tv.datasets.CIFAR10(root="./Cifar10/",train=False,download=True,transform=transformationTest)
    dataLoader = DataLoaderX(cifarDataTrainSet,batch_size=batchSize,shuffle=True,num_workers=4,pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### model construct
    model = EfficientReformModel.EfficientNetReform(in_channels=3,num_classes=labelsNumber,drop_connect_rate=0.25,w=w,d=d,classify=True).to(device)
    print(model)
    #lossCri = Model.LabelsSmoothingCrossLoss(labelsNumber,0.09).to(device)
    lossCri = nn.CrossEntropyLoss(reduction="mean").to(device)
    optimizer = rmsprop.RMSprop(model.parameters(),minLR,momentum=0.9,weight_decay=1e-5)
    if loadWeight :
        model.load_state_dict(torch.load(modelSavePath + "Model_" + str(trainModelLoad) + ".pth"))
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

    ### Train or Test
    scheduler = CosineDecaySchedule(lrMin=minLR,lrMax=maxLR,tMaxIni=tMaxIni,factor=1.15,lrDecayRate=decayRate,warmUpSteps=warmUpSteps)
    if ifTrain:
        model.train()
        trainingTimes = 0
        for e in range(epoch):
            for times , (images, labels) in enumerate(dataLoader):
                imagesCuda = images.float().to(device,non_blocking = True)
                labelsCuda = labels.long().to(device,non_blocking = True)
                predict = model(imagesCuda)
                oriLoss = lossCri(predict,labelsCuda)
                loss = oriLoss / stepTimes
                loss.backward()
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
                    #nn.utils.clip_grad_value_(model.parameters(), clip_value)
                    learning_rate = scheduler.calculateLearningRate()
                    state_dic = optimizer.state_dict()
                    state_dic["param_groups"][0]["lr"] = learning_rate
                    optimizer.load_state_dict(state_dic)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                if trainingTimes % saveTimes == 0 :
                    ### val part
                    model.eval()
                    predictList = []
                    truthList = []
                    k = 0
                    for testImage, testTarget in cifarDataTestSet:
                        predictTensor = model(testImage.unsqueeze(0).to(device)).cpu().detach().numpy()
                        position = np.argmax(np.squeeze(predictTensor))
                        if k % displayTimes == 0:
                            print("Validation : " + str(k))
                            print(position)
                            print(testTarget)
                        predictList.append(position)
                        truthList.append(testTarget)
                        k += 1
                        if k == valTestSampleNumber:
                            break
                    acc = metrics.accuracy_score(y_true=truthList, y_pred=predictList)
                    torch.save(model.state_dict(), modelSavePath + "Model_" + str(acc) + ".pth")
                    model.train()
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









