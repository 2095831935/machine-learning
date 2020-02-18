# 背景
**将约会对象分为三类：**
- 不喜欢的人
- 魅力一般的人
- 极具魅力的人

**每个约会对象有三个特征：**
- 每年获得的飞行常客里程数
- 玩视频游戏所耗时间百分比
- 每周消费的冰琪淋公升数
这些特征数据存放在文本文件datingTestSet.txt中；

**利用KNN算法对约会对象进行分类；算法的实现流程：**
1. 收集数据：提供文本文件。
2. 准备数据：使用Python解析文本文件。
3. 分析数据：使用Matplotlib画二维扩散图。
4. 训练算法：此步骤不适用于k近邻算法。
5. 测试算法：使用海伦提供的部分数据作为测试样本。
6. 测试样本和非测试样本的区别在于：测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。
7. 使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。

# 算法实现细节
``` python3
# 算法依赖的库
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
```

**step1**: 处理数据输入格式问题，创建名为`file2matrix`函数；
``` python3
def file2matrix(filename):
    """
    将文本记录转换到NumPy的解析程序
    
    注意Numpy库提供的数据操作并不支持Python自带的列表类型，因此在编写代码时要注意不要使用错误的数组类型；
    """
    fr = open(filename)
    arrayOlines = fr.readlines() 
    numberOfLines = len(arrayOlines) # 得到文件行数
    returnMat = zeros((numberOfLines, 3)) # 创建返回的Numpy矩阵
    classLabelVector = []
    index = 0
    # 解析文件数据到列表
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
```

**step2**: 分析数据：使用Matplotlib创建散点图；
``` python3
def imaging():
    """
    数据可视化
    """
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 
               15*array(datingLabels), 15*array(datingLabels))
    ax = fig.add_subplot(212)
    ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 
               15*array(datingLabels), 15*array(datingLabels))
    plt.legend()
    plt.show()
```

**step3**: 准备数据：归一化数值，将数据的值映射到0到1或-1到1之间；这是由于这三个特征的重要性是同等的，即权重相同；如果不归一化，显然在计算距离时，每年的飞行常客里程数的影响远远大于其它两个特征；
``` python3
def autoNorm(dataSet):
    """
    数据归一化
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet-tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1)) # 特征值相除
    return normDataSet, ranges, minVals
```

**step4**: 构造KNN分类器；`classify0`为蛮力实现方法；
```
def classify0(inX, dataSet, labels, k):
    """
    对未知类别属性的数据集中的每个点依次执行以下操作：
    1. 计算已知类别数据集中的点与当前点之间的距离；
    2. 按照距离递增次序排序；
    3. 选取与当前点距离最小的k个点；
    4. 确定前k个点所在类别的出现频率；
    5. 返回前k个点出现频率最高的类别作为当前点的预测分类。
    """
    dataSetSize = dataSet.shape[0]
    # 1.距离计算并排序
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 2.选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), 
                              # 3.排序
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```

**step5**: 测试算法：作为完整程序验证分类器；测试指标：分类错误率；其中90%数据用作训练集，10%数据用途测试集，由于数据本身就是随机的，因此直接用前10%的数据作为测试集；`k, hoRatio`是可以试错找到最优；
``` python3
def datingClassTest():
    """
    KNN算法性能测试
    """
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], \
                                     datingLabels[numTestVecs:m], 4)
        print("the classifier came back with: %d, the real answer is: %d" \
              % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
```
