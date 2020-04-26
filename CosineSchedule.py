import math

class CosineDecaySchedule :

    def __init__(self,lrMin,lrMax,tMaxIni,factor,lrDecayRate,warmUpSteps):
        """
        :param lrMin: The min learning rate in one schedule
        :param lrMax: The max learning rate in one schedule
        :param tMaxIni: The initial max training times in one schedule
        :param factor: increase tMaxIni by multiply this factor at every restart
        :param lrDecayRate : The decay rate of lrMax
        :param warmUpSteps : The warm up times.
        """
        self.lrMin = lrMin
        self.lrMax = lrMax
        self.curTT = 0.
        self.trainingTimes = tMaxIni
        self.factor = factor
        self.lrDecayRate = lrDecayRate
        self.warmUpSteps = warmUpSteps
        self.warmUpC = (lrMax - lrMin) / warmUpSteps
        self.ifWarmUp = True


    def calculateLearningRate(self):
        if self.ifWarmUp:
            return self.lrMin + self.warmUpC * self.curTT
        else:
            if self.curTT > self.trainingTimes:
                self.__restart()
            if self.lrMax > self.lrMin:
                lrC = float(self.lrMin) + 0.5 * (self.lrMax - self.lrMin) \
                      * (1. + math.cos(self.curTT / self.trainingTimes * math.pi))
                return lrC
            else:
                lrC = self.lrMin
                return lrC

    def step(self):
        self.curTT += 1
        if self.curTT == self.warmUpSteps and self.ifWarmUp is True:
            self.ifWarmUp = False
            self.curTT = 0

    def __restart(self):
        self.lrMax = self.lrMax * self.lrDecayRate
        self.curTT = 0
        self.trainingTimes = self.trainingTimes * self.factor
        self.ifWarmUp = False

if __name__ == "__main__":
    testSche = CosineDecaySchedule(2e-6,7e-4,1100,1.15,0.85,1500)
    testTotalTrainingTimes = 30000
    x = []
    for i in range(testTotalTrainingTimes):
        x.append(testSche.calculateLearningRate())
        testSche.step()
    import matplotlib.pyplot as  plt
    plt.figure()
    plt.plot([i for i in range(testTotalTrainingTimes)],x , lw = 2)
    plt.show()

