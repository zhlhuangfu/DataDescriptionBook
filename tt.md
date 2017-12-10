# Bag of Words Meets Bags of Popcorn

Link: [https://www.kaggle.com/c/word2vec-nlp-tutorial/](https://www.kaggle.com/c/word2vec-nlp-tutorial/)

### 

### 题目描述

对情感分析进行了更深入的研究。人们用语言来表达自己的情绪，这种语言经常被讽刺，模棱两可和言语所掩盖，所有这些对人类和计算机来说都是非常具有误导性的。而很多机器学习技术使得计算能力和准确性的提升成为可能。

在该问题中给出一句电影评论，判断该电影评论是积极的还是消极的。

采用的方法一般为：第一步从自然语言中提取出特征，第二步对特征进行适当的分类器训练。

第一步可以尝试采用Google的Word2Vec方法，Word2Vec试图理解单词之间的含义和语义关系。，它的工作方式类似于深度方法，如递归神经网络或深度神经网络，但计算效率更高。

### 

### 先修技能

* 掌握词袋法、TF-IDF或Word2Vec等文本特征提取技术。
* 掌握神经网络、朴素贝叶斯或其他分类器。
* 理解AUC评价指标。
* 会svm或者knn等分类器的使用。

### 

### 输入格式

* `labeledTrainData.csv`是有Label的训练集，包含20000行评论数据，包括评论id、评论文本、情感（0代表消极，1代表积极）。
* `testData.csv`是一个无Label的测试集，包含5000行评论数据，包括评论id、评论文本，没有对应情感。
* `unlabeledTrainData`是额外的50000行无Label数据，包括评论id和评论文本。**可用来进行文本特征提取或半监督学习。**
* `sampleSubmission`是一个提交格式的样例。

### 

### 输出格式

您的提交csv文件应包含行名，并采用以下格式：对于测试集中的每条评论，输出一行，其中包含评论id和对应预测的结果（**注意，可以提交\[0, 1\]以内的任意数值，表示预测为正例的score，最终会采用AUC进行评价，**[**AUC的科普链接**](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)）。如下所示：

```
id,sentiment
3862_4,0
674_10,1
8828_10,0

(4997 more lines)
```

### 

### 评价

使用 AUC 最后评判标准。

### 

### 代码与数据

* labeledTrainData: [https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/labeledTrainData.tsv](https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/labeledTrainData.tsv)
* testData: [https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/testData.tsv](https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/testData.tsv)
* unlabeledTrainData: [https://www.kaggle.com/c/word2vec-nlp-tutorial/download/unlabeledTrainData.tsv.zip](https://www.kaggle.com/c/word2vec-nlp-tutorial/download/unlabeledTrainData.tsv.zip)
* sampleSubmission: [https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/sampleSubmission.csv](https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/sampleSubmission.csv)
* correct\_submission: [https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/correct\_submission.csv](https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/correct_submission.csv)

### 

### 完整代码

* \(MultinomialNB version\) Accuracy = 0.85: [https://www.kaggle.com/ayanmaity/bag-of-words-meets-popcorns](https://www.kaggle.com/jiuzhang/jiuzhang-knn-sk-learn)
* 官方教程：[https://www.kaggle.com/c/word2vec-nlp-tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial)

### 

### 测评配置环境

python

```
pip install -U numpy
pip install pandas
pip install -U scikit-learn
```

测评代码

```py
from sklearn.metrics import roc_auc_score
import pandas as pd
y_test = pd.read_csv(data_dir + "correct_submission.csv") # 正确答案
y_pred = pd.read_csv(data_dir + "prediction_test.csv") # 用户预测的答案
auc = roc_auc_score(y_test['sentiment'],y_pred['sentiment'])
```



