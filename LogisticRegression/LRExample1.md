# 二维示例
&emsp;&emsp;有100个样本点，每个点包含两个数值型特征：X1和X2。在此数据集上，我们将通过使用梯度上升法找到最佳回归系数，也就是拟合出Logistic回归模型的最佳参数。

**step1** 梯度下降优化算法  
算法伪代码:  
&emsp;每个回归系数初始化为1  
&emsp;重复R次：  
&emsp;&emsp;计算整个数据集的梯度  
&emsp;&emsp;使用`alpha × gradient`更新回归系数的向量  
&emsp;&emsp;返回回归系数  
 ``` python3
 def loadDataSet():
    """
    输入：testSet.txt文件中的样本点；
    输出：样本矩阵及标签向量；
    """
    dataMat, labelMat = [], []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    """
    输出sigmoid函数在inX的值；
    """
    inX = np.matrix(inX)
    m,n = np.shape(inX)
    ans = np.zeros((m, 1))
    for i in range(m):
        ans[i][0] = 1.0 / (1+math.exp(-inX[i][0]))
    return ans

def gradAscent(dataMatIn, classLabels):
    """
    梯度下降找到最佳参数；
    输出为特征的权重系数；
    """
    # 首先转换为Numpy矩阵数据类型
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    
    m,n = np.shape(dataMatrix)
    alpha, maxCycles, weights = 0.001, 500, np.ones((n,1))
    for k in range(maxCycles):
        # 矩阵相乘
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights
 ```
 
 **step2** 画出决策边界；
 ``` python3
 def plotBestFit(wei):
    """
    画出散点图及分隔边界
    """
    # 将样本点按标签分别放在[xcord1, ycord1]和[xcord2, ycord2]中
    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1, ycord1, xcord2, ycord2 = [], [], [], []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    
    # 画出拟合曲线
    x = np.arange(-3.0, 3.0, 0.1)
    # 最佳拟合直线
    y = (-weights[0]-weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
 ```
 **step3** 梯度下降法只适用于小数据集，为了适应大数据集的优化算法，可以利用随机梯度下降算法来计算参数；
 ``` python3
 def stocGradAscent(dataMatrix, classLabels, numIter=150):
    """
    随机梯度上升；
    """
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            # alpha随着迭代次数增加而减少，但不是严格递减，避免参数的严格下降也常见于模拟退火算法等其它优化算法；
            alpha = 4.0 / (1+j+i) + 0.01
            # 随机选择一个样本进行参数更新；
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h[0][0]
            weights = weights + alpha*error*dataMatrix[randIndex]
    return weights
 ```
