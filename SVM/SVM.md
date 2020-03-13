# 介绍
用SVM对testSet.txt中的训练数据进行分类，其中smoSimple()是简化版的SMO求解算法，smoP()是完整版的SMO求解算法。

``` python3
import numpy as np
import math
import matplotlib.pyplot as plt

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2))) # 误差缓存，其中第一列给出的是eCache是否有效的标志位，而第二列给出的是实际的E值；
        
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b
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
        eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
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
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
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
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
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
    
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    创建一个alpha向量并将其初始化为0向量
    当迭代次数小于最大迭代次数时（外循环）
        对数据集中的每个数据向量（内循环）：
        如果该数据向量可以被优化：
            随机选择另外一个数据向量
            同时优化这两个向量
            如果两个向量都不能被优化，退出内循环
    如果所有向量都没被优化，增加迭代数目，继续下一次循环
    """
    dataMatrix, labelMat = np.mat(dataMatIn), np.mat(classLabels).transpose()
    b = 0; m, n =  np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))    # 初始化alpha值；
    iterIndex = 0
    while iterIndex < maxIter:
        alphaPairsChanged = 0
        for  i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i]) # fXi是预测值，Ei是误差值；
            # 如果alpha可以更改进入优化过程；
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                
                # 保证alpha在0与C之间；
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[i]+alphas[j]-C)
                    H = min(C, alphas[i]+alphas[j])
                if L==H:
                    # 如L==H，说明alpha[i]与alpha[j]都在边界上，不用修改；
                    print("L==H")
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    # 由于后面eta要用作除数，因此不能等于0；另外eta=-||dataMatrix[i,:]-dataMatrix[j,:]||^2肯定小于0，大于0是不可能的；
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                
                # 对i进行修改，修改量与j相同，但方向相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
                if 0 < b1 and b1 < C:
                    b = b1
                elif 0 < b2 and b2 < C:
                    b = b2
                else:
                    b = (b1+b2)/2
                alphaPairsChanged = 1
                print("iter: %d i: %d, pairs changed %d" % (iterIndex, i, alphaPairsChanged))
            if alphaPairsChanged == 0:
                iterIndex += 1
            else:
                iterIndex = 0
            print("iteration number: %d" % iterIndex)
        return b, alphas

def calcWs(alphas, dataArr, classLabels):
    X, labelMat = np.mat(dataArr), np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i], X[i,:].T)
    return w

def plotBestFit(w, b, dataMatrix, labelMatrix, alphas):
    """
    画出散点图及分隔边界
    """
    # 将样本点按标签分别放在[xcord1, ycord1]和[xcord2, ycord2]中
    n = np.shape(dataMatrix)[0]
    xcord1, ycord1, xcord2, ycord2, xcord3, ycord3 = [], [], [], [], [], []
    for i in range(n):
        if int(labelMatrix[i]) == 1:
            xcord1.append(dataMatrix[i,0])
            ycord1.append(dataMatrix[i,1])
        else:
            xcord2.append(dataMatrix[i,0])
            ycord2.append(dataMatrix[i,1])
        if alphas[i] > 0.0:
            xcord3.append(dataMatrix[i,0])
            ycord3.append(dataMatrix[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    ax.scatter(xcord3, ycord3, s=20, c='black')
    # 画出拟合曲线
    x = np.arange(2.0, 6.5, 0.1)
    # 最佳拟合直线
    y = (-b.A-w[0]*x) / w[1]
    ax.plot(x, y.T)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
```
