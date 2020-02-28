# 从个人广告中获取区域倾向
&emsp;&emsp;将分别从美国的两个城市中选取一些人，通过分析这些人发布的征婚广告信息，来比较这两个城市的人们在广告用词上是否不同。如果结论确实是不同，那么他们各自常用的词是哪些？从人们的用词当中，我们能否对不同城市的人所关心的内容有所了解？

## 算法框架
1. 收集数据：从RSS源收集内容，这里需要对RSS源构建一个接口。
2. 准备数据：将文本文件解析成词条向量。
3. 分析数据：检查词条确保解析的正确性。
4. 训练算法：使用我们之前建立的trainNB0()函数。
5. 测试算法：观察错误率，确保分类器可用。可以修改切分程序，以降低错误率，提高分类结果。
6. 使用算法：构建一个完整的程序，封装所有内容。给定两个RSS源，该程序会显示最常用的公共词。

## 代码实现
**step1** 收集数据：导入RSS源；
``` python3
ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
sf = feedparser.parse('http://rss.yule.sohu.com/rss/yuletoutiao.xml')
```

**step2** RSS源分类器及高频词去除函数；
- 筛选出高频词；
``` python3
def calcMostFreq(vocabList, fullText):
    """
    计算vocabList中的高频词
    """
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]
```

- 训练分类器并进行测试；
``` python3
def localWords(feed1, feed0):
    """
    训练分类器并进行测试
    """
    docList, classList, fullText = [], [], []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        # 每次访问一条RSS源
        # 标签为1的RSS源
        wordList = textSparse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # 标签为0的RSS源
        wordList = textSparse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 从vocabList中去掉出现次数最高的那些词
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    # 随机选出20个测试样本；其它样本作为训练样本；
    trainingSet, testSet = list(range(2*minLen)), []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 训练测试
    p0V, p1V, pSpam = trainNB1(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is: ", float(errorCount) / len(testSet))
    return vocabList, p0V, p1V
```

- 显示地域相关的用词
``` python3
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY **"
    for item in sortedNY:
        print item[0]
```

# 补充
&emsp;&emsp;这里两个RSS源可能有问题；
