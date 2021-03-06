<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 一、朴素贝叶斯
&emsp;&emsp;朴素贝叶斯 (NB) 算法是生成模型，给出一个最优的猜测结果，同时给出这个结果的概率估计值；朴素贝叶斯之所以称为“朴素”，是因为该算法的前提假设是最原始的、最简单的。本文首先介绍朴素贝叶斯算法的原理，其次给出其实现代码，最后给出朴素贝叶斯算法的实例；

## 1.1 背景知识
- 贝叶斯决策理论  
&emsp;&emsp;若一个样本属于类别1的概率为p1，属于类别2的概率为p2，当p1>p2，那么猜测该样本的类别为1；当p2>p1，那么猜测该样本的类别为2；

- 条件概率  
&emsp;&emsp;如果已知$P(x|c)$，则$P(c|x)=\frac{P(x|c)P(c)} {P(X)}$;

## 算法原理
&emsp;&emsp;朴素贝叶斯算法的两个前提假设：**1**. 样本特征之间相互独立；**2**. 每个特征同等重要；尽管这两个假设存在一些瑕疵，但朴素贝叶斯的实际效果却很好；

## 算法框架

## 算法分析

**优点**
- 朴素贝叶斯模型发源于古典数学理论，有稳定的分类效率。
- 对小规模的数据表现很好，能个处理多分类任务，适合增量式训练，尤其是数据量超出内存时，我们可以一批批的去增量训练。
- 对缺失数据不太敏感，算法也比较简单，常用于文本分类。

**缺点**
- 理论上，朴素贝叶斯模型与其他分类方法相比具有最小的误差率。但是实际上并非总是如此，这是因为朴素贝叶斯模型给定输出类别的情况下,假设属性之间相互独立，这个假设在实际应用中往往是不成立的，在属性个数比较多或者属性之间相关性较大时，分类效果不好。而在属性相关性较小时，朴素贝叶斯性能最为良好。对于这一点，有半朴素贝叶斯之类的算法通过考虑部分关联性适度改进。
- 需要知道先验概率，且先验概率很多时候取决于假设，假设的模型可以有很多种，因此在某些时候会由于假设的先验模型的原因导致预测效果不佳。
- 由于我们是通过先验和数据来决定后验的概率从而决定分类，所以分类决策存在一定的错误率。
- 对输入数据的表达形式很敏感。



## 代码实现

## 应用实例
- NB进行文本分类（textCategorization.md）；
- NB过滤垃圾邮件；
- 用NB从个人广告中获取区域倾向；

