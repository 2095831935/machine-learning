# 决策树的实现细节

``` python3
# 算法依赖的库
from math import log
import operator
import matplotlib.pyplot as plt
```

**step1** 计算给定数据集的香农熵；
``` python3
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        # 以2为底求对数
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
```
**step2** 划分数据集；
1. 按照给定特征划分数据集；
``` python3
def splitDataSet(dataSet, axis, value):
    """
    注意计算条件熵时，需要计算相应特征下不同值的熵（该特征为离散特征）；
    该函数表示根据第axis个特征的值value划分数据集，返回dataSet中所有第i个特征==value的样本；
    """
    # 创建新的List对象
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 抽取
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
```
2. 选择最好的特征划分数据集，即信息增益最大的特征；
``` python3
def chooseBestFeatureToSplit(dataSet):
    """"
    返回信息增益最大对应的特征；
    """"
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计算每种划分方式 信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            # 计算最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
```

**step3** 递归构建决策树；归结束条件：程序遍历完所有划分数据集的属性，或者每个分支下的所有实例都具有相同的分类。
1. 如果使用了所有的属性，但是叶结点的类标签依然不是唯一，通常使用多数表决的方法决定该叶节点的分类。
``` python3
def majorityCnt(classList):
    """
    classList中包含的是叶结点样本的类标签；
    返回多数表决的结果；
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```
2. 用递归创建树的结构；
``` python3
def createTree(dataSet, labels):
    """
    返回的是树的字典结构
    """
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 还有特征没有处理时选择最优属性进行划分
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    # 得到列表包含的所属属性值
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
```
**step4** 使用Matplotlib绘制树形图;
1. 绘制树节点；
``` python3
# 使用文本注解绘制树节点
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
# 绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    使用文本注解绘制树节点
    """
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

def createPlot():
    """
    简单示例
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode("决策节点", (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode("叶节点", (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()
```
2. 绘制整棵树；
    - 确定树有多少叶节点及多少层，从而确定x轴与y轴的长度；
``` python3
def getNumLeafs(myTree):
    """
    为了确定x轴的长度，需要确定树有多少个叶节点；
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试节点的数据类型是否为字典
        if type(secondDict[key]).__name__ == "dict":
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    """
    为了确定y轴的长度，需要确定树有多少层；
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0] # 注意python3的keys()方法不再返回列表，而是"dict_keys"对象，不支持索引；
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth        
```
- 绘制决策树；注意树的根结点为当前树所有叶节点的中间位置；这里重新定义了`createPlot`函数；
``` python3
def plotMidText(cntrPt, parentPt, txtString):
    """
    在父子节点间填充文本信息
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid*(1+0.03), yMid, txtString, va="center", ha="center", rotation=0)

def plotTree(myTree, parentPt, nodeTxt):
    """
    parentPt为myTree的父节点位置；
    (plotTree.xoff, plotTree.yoff)为上一个叶子节点的位置；
    """
    # 计算宽和高
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    # 将当前根节点置入myTree所有叶节点的中间
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs) ) / 2.0 / plotTree.totalW, plotTree.yoff )
    # 标记子节点属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    # 减小y偏移
    plotTree.yoff = plotTree.yoff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)  
            plotMidText((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    # 递归结束，还原y偏移
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD
    
 def createPlot(inTree):
    """
    简单示例
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
```

**step5** 测试和存储分类器；
    - 使用决策树执行分类；
