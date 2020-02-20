# 手写识别系统

## 问题背景
构造K近邻分类器的手写识别系统，该系统只能识别数字0到数字9，并将输入数据转换成具有相同的色彩和大小：宽高是32像素x32像素的黑白图像；

## 数据说明
trainingDigits中包含的是训练数据，testDigits中包含的是测试数据；

## 构造框架
1. 收集数据：提供文本文件。
2. 准备数据：编写函数classify0()，将图像格式转换为分类器使用的list格式。
3. 分析数据：在Python命令提示符中检查数据，确保它符合要求。
4. 训练算法：此步骤不适用于k近邻算法。
5. 测试算法：编写函数使用提供的部分数据集作为测试样本，测试样本与非测试样本的区别在于测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。
6. 使用算法：本例没有完成此步骤，若你感兴趣可以构建完整的应用程序，从图像中提取数字，并完成数字识别，美国的邮件分拣系统就是一个实际运行的类似系统。

## 算法实现细节
``` python3
# 框架依赖的包
from os import listdir
```
**step1** 准备数据：将图像转换为测试向量;
``` python3
def img2vector(filename):
    """
    将32x32的像素矩阵转换成1x1024的向量
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect
```

**step2** 测试算法：使用用K近邻算法识别手写数字；
``` python3
def handwritingClassTest():
    """
    手写数字识别系统的测试代码
    """
    hwLabels = []
    trainingFileList = listdir(r'.\digits\trainingDigits') # 获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 从文件名解析分类数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(r'.\digits\trainingDigits\%s'% fileNameStr)
    testFileList = listdir(r'.\digits\testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(r'.\digits\testDigits\%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 4)
        print('the classifier came back with: %d, the real answer is: %d'%(classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total number error rate is: %f" % (errorCount/float(mTest)))
```
注意：其中的某些函数是源自`data website.md`文件；
