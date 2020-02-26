# 垃圾邮件过滤
&emsp;&emsp;使用朴素贝叶斯解决一些现实生活中的问题时，需要先从文本内容得到字符串列表，然后生成词向量。本例中，我们将了解朴素贝叶斯的一个最著名的应用：电子邮件垃圾过滤。

## 算法框架
1. 收集数据：提供文本文件。
2. 准备数据：将文本文件解析成词条向量。
3. 分析数据：检查词条确保解析的正确性。
4. 训练算法：使用我们之前建立的trainNB0()函数。
5. 测试算法：使用classifyNB()，并且构建一个新的测试函数来计算文档集的错误率。
6. 使用算法：构建一个完整的程序对一组文档进行分类，将错分的文档输出到屏幕上。

## 代码实现
``` python3
# 算法依赖的库
import re
```
**step1** 准备数据：切分文本；
``` python3
def textSparse(bigString):
    """
    正则表达式进行文本解析；
    """
    listOfTokens = re.split(r'\w*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
```

**step2** 邮件分类；其中NB的训练等相关函数源自textCategorization.md文件中；
``` python3

```
