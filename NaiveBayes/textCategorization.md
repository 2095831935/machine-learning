# 示例：用NB进行文本分类
&emsp;&emsp;以在线社区的留言板为例。为了不影响社区的发展，我们要屏蔽侮辱性的言论，所以要构建一个快速过滤器，如果某条留言使用了负面或者侮辱性的语言，那么就将该留言标识为内容不当。过滤这类内容是一个很常见的需求。对此问题建立两个类别：侮辱类和非侮辱类，使用1和0分别表示。

``` python3
# 依赖的库
from numpy import *
``` 

**step1** 根据文本构建特征向量  
&emsp;&emsp;利用词袋模型来构建特征向量，即考虑出现在所有文档中的所有单词，再决定将哪些词纳入词汇表或者说所要的词汇集合（词袋），然后必须要将每一篇文档转换为词汇表上的向量。
``` python3
def loadDataSet():
    """
    创建一些实例样本用来
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
            ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
            ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
            ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
            ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
            ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
            ]
    # 1代表侮辱性文字，0代表正常言论；
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):
    """
    构建词袋； 
    """
    # 创建一个空集；
    vocabSet = set()
    for document in dataSet:
        # 创建两个集合的并集；
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    基于词袋，将文档转换成词向量；
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    """
    该词袋模型是对setOfWords2Vec的修正；
    在词袋中，每个单词可以出现多次，而在词集中，每个词只能出现一次。
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec
```

**step2** 从词向量计算概率;  
``` python3
def trainNB0(trainMatrix, trainCategory):
    """
    trainMatrix: n*m，词向量；
    trainCategory: 1*n，类别标签；
    计算条件概率；
    """
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 初始化概率
    p0Num, p1Num = zeros(numWords), zeros(numWords)
    p0Denom, p1Denom = 0, 0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 向量相加
            p1Num += trainMatrix[i]
            p1Denom += 1
        else:
            p0Num += trainMatrix[i]
            p0Denom += 1
    # 对每个元素做除法计算条件概率；
    p1Vect1 = p1Num / p1Denom
    p1Vect0 = ones(ones(numWords) - p1Vect1)
    p0Vect1 = p0Num / p0Denom
    p0Vect0 = ones(ones(numWords) - p0Vect1)
    return p0Vect0, p0Vect1, p1Vect0, p1Vect1, pAbusive

def trainNB1(trainMatrix, trainCategory):
    """
    trainMatrix: n*m，词向量；
    trainCategory: 1*n，类别标签；
    计算条件概率；
    考虑到拉普拉斯平滑和数据下溢的可能；
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 与trainNB0不同的是，初始化值不同；
    # 这是考虑了拉普拉斯平滑；其中p0Denom,p1Denom初始化为2是
    # 因为每个特征中有两种取值可能，即0或1；
    p0Num, p1Num = ones(numWords), ones(numWords)
    p0Denom, p1Denom = 2, 2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 向量相加
            p1Num += trainMatrix[i]
            p1Denom += 1
        else:
            p0Num += trainMatrix[i]
            p0Denom += 1
    # 与trainNB0不同的是，考虑到条件概率相乘可能会下溢
    # 因此将其全部转换成log值；
    p1Vect1 = log(p1Num / p1Denom) 
    p1Vect0 = log(ones(numWords) - p1Num / p1Denom)
    p0Vect1 = log(p0Num / p0Denom)
    p0Vect0 = log(ones(numWords) - p0Num / p0Denom)
    return p0Vect0, p0Vect1, p1Vect0, p1Vect1, pAbusive
```

**step3** 构建分类算法并进行测试；
``` python3
def classifyNB(vec2Classify, p0Vect0, p0Vect1, p1Vect0, p1Vect1, pClass1):
    """
    利用NB对样本vec2Classify进行分类；
    """
    p1 = sum(vec2Classify * p1Vect1) + sum((ones(len(vec2Classify)) - vec2Classify) * p1Vect0) + log(pClass1)
    p0 = sum(vec2Classify * p0Vect1) + sum((ones(len(vec2Classify)) - vec2Classify) * p0Vect0) + log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0Vect0, p0Vect1, p1Vect0, p1Vect1, pAbusive = trainNB1(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "'classified as: '", classifyNB(thisDoc, p0Vect0, p0Vect1, p1Vect0, p1Vect1, pAbusive))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "'classified as: '", classifyNB(thisDoc, p0Vect0, p0Vect1, p1Vect0, p1Vect1, pAbusive))
```


# 补充
&emsp;&emsp;本例中的训练代码部分与《机器学习实战》一书中的源码有所区别，主要在于计算条件概率和进行分类部分；（个人觉得书中计算条件概率时书中源码对分母的计算有问题，另外在分类时只考虑特征值为1的特征，为此，我根据西瓜书中提供的算法思路进行修正。）
