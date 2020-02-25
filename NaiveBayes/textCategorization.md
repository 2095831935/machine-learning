# 示例：用NB进行文本分类
&emsp;&emsp;以在线社区的留言板为例。为了不影响社区的发展，我们要屏蔽侮辱性的言论，所以要构建一个快速过滤器，如果某条留言使用了负面或者侮辱性的语言，那么就将该留言标识为内容不当。过滤这类内容是一个很常见的需求。对此问题建立两个类别：侮辱类和非侮辱类，使用1和0分别表示。

## 根据文本构建特征向量
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
```

## 训练：从词向量计算概率；
