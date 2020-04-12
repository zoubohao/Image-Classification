import torch
import torch.nn as nn
import torch.nn.functional as F


class SimulateCenterLoss(nn.Module):

    def __init__(self,labelsNumber,features,lambda1,weight = None):
        super(SimulateCenterLoss,self).__init__()
        self.classCenter = torch.nn.init.xavier_normal_(nn.Parameter(torch.randn(size=[labelsNumber,features],requires_grad=True),requires_grad=True))
        self.classifyLoss = nn.CrossEntropyLoss(weight=weight,reduction="sum")
        self.lambda1 = lambda1

    def forward(self,predictions,x,labels):
        """
        :param predictions: The shape of predictions is [batch, labels]
        :param labels: [batch]
        :param x : [batch,features]
        :return:
        """
        loss1 = self.classifyLoss(predictions,labels)
        correspondingMatrix = F.embedding(labels,self.classCenter,padding_idx=0,scale_grad_by_freq=True)
        loss2 = self.lambda1 / 2. * torch.pow(torch.sum(torch.sub(x,correspondingMatrix)),2.0)
        return loss1 + loss2



if __name__ == "__main__":
    import numpy as np
    testLoss = SimulateCenterLoss(10,512,0.5)
    testPredictions = torch.randn(size=[4,10],dtype=torch.float32)
    testLabels = torch.from_numpy(np.array([1,0,2,3])).long()
    testX =  torch.randn(size=[4,512],dtype=torch.float32).float()
    print(testLoss(testPredictions,testX,testLabels))