from Models.MobileNetV2 import MobileNetV2
from Models.ResNest import ResNestNet
import torchvision as tv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np




class MetaClassify(nn.Module):

    def __init__(self,fea_num):
        super().__init__()
        self.mbLinear = nn.Linear(fea_num,1024,bias=True)
        self.resLinear = nn.Linear(fea_num,1024,bias=True)
        self.totalLinear = nn.Linear(1024,10,bias=True)


    def forward(self,x1,x2):
        medium = self.mbLinear(x1) + self.resLinear(x2)
        return self.totalLinear(medium)


class MetaDataSet(Dataset):

    def __init__(self,mbPath,resPath):
        super().__init__()
        self.dataMB = []
        self.dataRes = []
        self.labels = []
        with open(mbPath,"r") as rh:
            for oneLine in rh:
                thisLine = oneLine.strip("\n")
                dataPart = thisLine.split(",")[0].split("\t")[0:-1]
                labelPart = thisLine.split(",")[1]
                self.labels.append(float(labelPart))
                thisData = []
                for string in dataPart:
                    #print(string)
                    thisData.append(float(string))
                self.dataMB.append(np.array(thisData))

        with open(resPath,"r") as rh:
            for oneLine in rh:
                thisLine = oneLine.strip("\n")
                dataPart = thisLine.split(",")[0].split("\t")[0:-1]
                thisData = []
                for string in dataPart:
                    thisData.append(float(string))
                self.dataRes.append(np.array(thisData))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ## Mb ,  Res , label
        return torch.as_tensor(self.dataMB[index],dtype=torch.float32), \
               torch.as_tensor(self.dataRes[index],dtype=torch.float32), \
               torch.as_tensor(self.labels[index],dtype=torch.long)



if __name__ == "__main__":

    ## config 0.9351
    labelsNumber = 10
    ifCNewData = False

    ifTMetaModel = False
    epoch = 200
    displayTimes = 100
    saveTimes = 2000
    lr = 1e-6
    weight_decay = 1.5e-6

    ifFTest = True
    metaLoadWeight = "./Model_Weight/Model_Meta_0.9351.pth"

    ##
    modelRes = ResNestNet(3, labelsNumber, 0.47, 1, 2)
    modelMB = MobileNetV2(3, labelsNumber, 0.45, 2, 2)
    modelRes.load_state_dict(torch.load("./Model_Weight/Model_Res0.9051.pth"))
    modelMB.load_state_dict(torch.load("./Model_Weight/Model_MB0.9258.pth"))
    transformation = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    modelRes = modelRes.eval()
    modelMB = modelMB.eval()
    ### Construct new training data set.
    if ifCNewData :
        ### Training feature
        newX_MB = list()
        newX_Res = list()
        newY = list()
        cifarDataTrainSet = tv.datasets.CIFAR10(root="./Cifar10/", train=True, download=True, transform=transformation)
        trainloader = DataLoader(cifarDataTrainSet, batch_size=100, shuffle=True)
        for idx, (batch_image, batch_labels) in enumerate(trainloader):
            print(idx)
            outputMB = modelMB(batch_image).detach().numpy()
            outputRes = modelRes(batch_image).detach().numpy()
            newX_MB.append(outputMB)
            newX_Res.append(outputRes)
            newY.append(batch_labels.detach().numpy())
            # if idx == 10 :
            #     break
        newX_Res = np.array(newX_Res).reshape([-1,10])
        newX_MB = np.array(newX_MB).reshape([-1,10])
        newY =  np.array(newY).reshape([-1])
        with open("./ResFeature.txt",mode="w") as wh:
            for i , resFea in enumerate(newX_Res):
                for value in resFea:
                    wh.write(str(value) + "\t")
                wh.write(",")
                wh.write(str(newY[i]))
                wh.write("\n")
        with open("./MBFeature.txt",mode="w") as wh:
            for i , mbFea in enumerate(newX_MB):
                for value in mbFea:
                    wh.write(str(value) + "\t")
                wh.write(",")
                wh.write(str(newY[i]))
                wh.write("\n")
        ### Testing feature
        newX_MB = list()
        newX_Res = list()
        newY = list()
        cifarDataTestSet = tv.datasets.CIFAR10(root="./Cifar10/", train=False, download=True, transform=transformation)
        testLoader = DataLoader(cifarDataTestSet, batch_size=100, shuffle=True)
        for idx, (batch_image, batch_labels) in enumerate(testLoader):
            print(idx)
            outputMB = modelMB(batch_image).detach().numpy()
            outputRes = modelRes(batch_image).detach().numpy()
            newX_MB.append(outputMB)
            newX_Res.append(outputRes)
            newY.append(batch_labels.detach().numpy())
            # if idx == 10 :
            #     break
        newX_Res = np.array(newX_Res).reshape([-1,10])
        newX_MB = np.array(newX_MB).reshape([-1,10])
        newY =  np.array(newY).reshape([-1])
        with open("./ResFeatureTest.txt",mode="w") as wh:
            for i , resFea in enumerate(newX_Res):
                for value in resFea:
                    wh.write(str(value) + "\t")
                wh.write(",")
                wh.write(str(newY[i]))
                wh.write("\n")
        with open("./MBFeatureTest.txt",mode="w") as wh:
            for i , mbFea in enumerate(newX_MB):
                for value in mbFea:
                    wh.write(str(value) + "\t")
                wh.write(",")
                wh.write(str(newY[i]))
                wh.write("\n")

    ### Training meta model.
    if ifTMetaModel :
        ### Data set
        metaTrainSet = MetaDataSet("./MBFeature.txt","./ResFeature.txt")
        metaTrainLoader = DataLoader(metaTrainSet,batch_size=128,shuffle=True)
        metaTestSet = MetaDataSet("./MBFeatureTest.txt", "./ResFeatureTest.txt")
        metaTestLoader = DataLoader(metaTestSet, batch_size=128, shuffle=True)
        ### model
        meta_model = MetaClassify(10).to("cuda:0")
        lossCri = nn.CrossEntropyLoss(reduction="sum").to("cuda:0")
        optimizer = torch.optim.SGD(meta_model.parameters(),lr = lr, momentum=0.9,nesterov=True,weight_decay=weight_decay)
        #print(list(meta_model.parameters()))
        trainingTimes = 0
        meta_model = meta_model.train(True)
        for e in range(epoch):
            for idx ,(mbFeas, resFeas, targets) in enumerate(metaTrainLoader):
                optimizer.zero_grad()
                mbCuda = mbFeas.to("cuda:0")
                resCuda = resFeas.to("cuda:0")
                tCuda = targets.to("cuda:0")
                predict = meta_model(mbCuda,resCuda)
                criLoss = lossCri(predict,tCuda)
                criLoss.backward()
                if trainingTimes % displayTimes == 0:
                    with torch.no_grad():
                        _, predicted = predict.max(1)
                        total = tCuda.size(0)
                        correct = predicted.eq(tCuda).sum().item()
                        print("######################")
                        print("Epoch : %d , Training time : %d" % (e, trainingTimes))
                        print("Cri Loss is %.3f " % (criLoss.item()))
                        print("Learning rate is ", optimizer.state_dict()['param_groups'][0]["lr"])
                        print("Correct ratio is %3f "% (correct / total + 0.))
                        print("predicted labels : ",predict[0:2,])
                        print("Truth labels : ",tCuda[0:2,])
                optimizer.step()
                trainingTimes += 1
                if trainingTimes % saveTimes == 0:
                    ### val part
                    meta_model = meta_model.eval()
                    test_loss = 0
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for batch_idx, (testMB, testRes, testLabels) in enumerate(metaTestLoader):
                            testMBCuda = testMB.to("cuda:0")
                            testResCuda = testRes.to("cuda:0")
                            testLabelsCuda = testLabels.to("cuda:0")
                            outputs = meta_model(testMBCuda,testResCuda)
                            loss = lossCri(outputs, testLabelsCuda)
                            test_loss += loss.item()
                            _, predicted = outputs.max(1)
                            total += testLabelsCuda.size(0)
                            correct += predicted.eq(testLabelsCuda).sum().item()
                            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                    acc =  correct / total
                    torch.save(meta_model.state_dict(), "./Model_Weight/Model_Meta_" + str(acc) + ".pth")
                    meta_model = meta_model.train(mode=True)
    ### Final test
    if ifFTest:
        meta_model = MetaClassify(10).to("cuda:0")
        meta_model.load_state_dict(torch.load(metaLoadWeight))
        meta_model = meta_model.eval()
        metaTestSet = MetaDataSet("./MBFeatureTest.txt", "./ResFeatureTest.txt")
        metaTestLoader = DataLoader(metaTestSet, batch_size=128, shuffle=True)
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (testMB, testRes, testLabels) in enumerate(metaTestLoader):
                testMBCuda = testMB.to("cuda:0")
                testResCuda = testRes.to("cuda:0")
                testLabelsCuda = testLabels.to("cuda:0")
                outputs = meta_model(testMBCuda, testResCuda)
                _, predicted = outputs.max(1)
                total += testLabelsCuda.size(0)
                correct += predicted.eq(testLabelsCuda).sum().item()
                print(' Acc: %.3f%% (%d/%d)'% (100. * correct / total, correct, total))
        acc = correct / total



