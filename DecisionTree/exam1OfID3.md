# 一、背景
&emsp;&emsp;通过示例讲解决策树如何预测患者需要佩戴的隐形眼镜类型。
## 1.1 算法框架
1. 收集数据：提供的文本文件。
2. 准备数据：解析tab键分隔的数据行。
3. 分析数据：快速检查数据，确保正确地解析数据内容，使用createPlot()函数绘制最终的树形图。
4. 训练算法：使用3.1节的createTree()函数。
5. 测试算法：编写测试函数验证决策树可以正确分类给定的数据实例。
6. 使用算法：存储树的数据结构，以便下次使用时无需重新构造树。
## 1.2 数据介绍
&emsp;&emsp;隐形眼镜数据集是非常著名的数据集，它包含很多患者眼部状况的观察条件以及医生推荐的隐形眼镜类型。数据来源于UCI数据库，为了更容易显示数据，本例对数据做了简单的更改，数据存储在lenses.txt文本文件中。
## 1.3 创建决策树并画图表示
&emsp;&emsp;其中调用的函数全部源自ID3_DecisionTree.md文件中；
``` python3
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
print(lensesTree)
createPlot(lensesTree)
```
