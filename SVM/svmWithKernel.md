# 介绍
考虑核技巧时的SVM，这里只是RBF核函数可用。

``` python3
import numpy as np
import matplotlib.pyplot as plt

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2))) # 误差缓存，其中第一列给出的是eCache是否有效的标志位，而第二列给出的是实际的E值。
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
            
        
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    """
    内循环中的启发式方法
    """
    maxK, maxDeltaE, Ej = -1, 0, 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei-Ek)
            if deltaE > maxDeltaE:
                maxK, maxDeltaE, Ej = k, deltaE, Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej
    
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    """
    内循环
    """
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 第2个alpha的选择
        j, Ej = selectJ(i, oS, Ei)
        alphaIold, alphaJold = oS.alphas[i].copy(), oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold)<0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if 0 < oS.alphas[i] and oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] and oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1+b2)/2
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iterIndex = 0
    entireSet, alphaPairsChanged = True, 0
    while iterIndex < maxIter and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged = 0
        # 遍历所有值
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print("fullSet, iter:%d i:%d, pairs changed %d"%(iterIndex, i, alphaPairsChanged))
            iterIndex += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter:%d i:%d, pairs changed %d"%(iterIndex, i, alphaPairsChanged))
            iterIndex += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged==0:
            entireSet = True
        print("iteration number: %d"%iterIndex)
    return oS.b, oS.alphas
                
def loadDataSet(fileName):
    """
    将文件fileName中的数据导入dataMat和labelMat中；
    """
    dataMat, labelMat = [], []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i,m):
    """
    选择第j个变量，保证第j个变量与第i个变量不重复；
    """
    j = i
    while j==i:
        j = int(np.random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    """
    根据约束L<=aj<=H，对aj进行剪枝；
    """
    if aj > H:
        return H
    elif aj < L:
        return L
    else:
        return aj
        
def kernelTrans(X, A, kTup):
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin':
        K = X*A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        # 元素间的除法
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError("Houston We Have a Problem - That Kernel is not recognized")
    return K

def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet("testSetRBF.txt")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf',k1))
    dataMat, labelMat = np.mat(dataArr), np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    # 构建支持向量矩阵
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m,n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataMat, labelMat = np.mat(dataArr), np.mat(labelArr).transpose()
    m,n = np.shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))

if __name__ == "__main__":
    testRbf(1.3)
```
