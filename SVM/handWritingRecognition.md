# 介绍
用SVM对digits文件中的手写数字进行分类，其中用到的函数引用自svmWithKernel.md。
``` python3
def img2vector(filename):
    """
    将32x32的像素矩阵转换成1x1024的向量
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName) # 获取目录内容
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        # 从文件名解析分类数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector(r'%s\%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages(r'.\digits\trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat, labelMat = np.mat(dataArr), np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    # 构建支持向量矩阵
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m,n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr, labelArr = loadImages(r'.\digits\testDigits')
    errorCount = 0
    dataMat, labelMat = np.mat(dataArr), np.mat(labelArr).transpose()
    m,n = np.shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))
```
