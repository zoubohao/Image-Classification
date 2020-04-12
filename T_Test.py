import scipy.stats as stats
import numpy as np

### Type I errors happen when we reject a true null hypothesis
### Type II errors happen when we fail to reject a false null hypothesis

### Typically when we try to decrease the probability one type of error, the probability for the other type increases.
### We could decrease the value of alpha from 0.05 to 0.01, corresponding to a 99% level of confidence.
### However, if everything else remains the same, then the probability of a type II error will nearly always increase.


### At this experiment, we only test the Type I error. Because in the
### real world, we care about the Type I error. So, the Null hypothesis must be true.
def oneExperimentTest(sd1, sd2, samplesNumber):
    x1 = np.random.normal(loc=0.,scale=sd1,size=samplesNumber // 2)
    x2 = np.random.normal(loc=0.,scale=sd2,size=samplesNumber)
    _, p = stats.ttest_ind(x1,x2,equal_var=True)
    return p

def oneEstimate(sd1,sd2,sampleNumber,testingTimes,significantLevel):
    errorTotalTimes = 0.
    for i in range(testingTimes):
        pValue = oneExperimentTest(sd1, sd2, sampleNumber)
        if i % 1000. == 0:
            print(str(i) + " : " + str(pValue))
        if pValue <= significantLevel:
            errorTotalTimes += 1.
    return errorTotalTimes / float(testingTimes)

def generateData(generateDataNumber,sampleNumber = 1000, testingTimes = 1000000, significantLevel = 0.05):
    fixedSd1 = 1.7
    floatSd2 = 2.3
    sdDistance = []
    errorRate = []
    for g in range(generateDataNumber):
        oneGResult = oneEstimate(fixedSd1,floatSd2,sampleNumber,testingTimes,significantLevel)
        print(oneGResult)
        sdDistance.append(floatSd2 - fixedSd1)
        errorRate.append(oneGResult)
        floatSd2 = floatSd2 + 0.6
    return sdDistance, errorRate



if __name__ == "__main__":
    ### config
    ### we should use big sample size, like 1000000 (yi bai wan)
    testSdDis , testErrorRate = generateData(10,significantLevel=0.05)
    print(testSdDis)
    print(testErrorRate)


