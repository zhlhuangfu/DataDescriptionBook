# Bag of Words Meets Bags of Popcorn

## 各种各样的电影评论，有讽刺、有反语，你能识别出这些评论是在褒奖还是在批评吗？

Link: [https://www.kaggle.com/c/word2vec-nlp-tutorial/](https://www.kaggle.com/c/word2vec-nlp-tutorial/)

### 题目描述

对情感分析进行了更深入的研究。人们用语言来表达自己的情绪，这种语言经常被讽刺，模棱两可和言语所掩盖，所有这些对人类和计算机来说都是非常具有误导性的。而很多机器学习技术使得计算能力和准确性的提升成为可能。

在该问题中给出一句电影评论，判断该电影评论是积极的还是消极的。

采用的方法一般为：第一步从自然语言中提取出特征，第二步对特征进行适当的分类器训练。

第一步可以尝试采用Google的Word2Vec方法，Word2Vec试图理解单词之间的含义和语义关系。它的工作方式类似于深度方法，如递归神经网络或深度神经网络，但计算效率更高。

### 先修技能

* 掌握词袋法、TF-IDF或Word2Vec等文本特征提取技术。
* 掌握神经网络、朴素贝叶斯或其他分类器。
* 理解AUC评价指标。
* 会svm或者knn等分类器的使用。

### 输入格式

* `labeledTrainData.csv`是有Label的训练集，包含20000行评论数据，包括评论id、评论文本、情感（0代表消极，1代表积极）。
* `testData.csv`是一个无Label的测试集，包含5000行评论数据，包括评论id、评论文本，没有对应情感。
* `unlabeledTrainData`是额外的50000行无Label数据，包括评论id和评论文本。**可用来进行文本特征提取或半监督学习。**
* `sampleSubmission`是一个提交格式的样例。

### 输出格式

您的提交csv文件应包含行名，并采用以下格式：对于测试集中的每条评论，输出一行，其中包含评论id和对应预测的结果（**注意，可以提交\[0, 1\]以内的任意数值，表示预测为正例的score，最终会采用AUC进行评价，**[**AUC的科普链接**](https://baike.baidu.com/item/AUC/19282953?fr=aladdin)）。如下所示：

```
id,sentiment
3862_4,0
674_10,1
8828_10,0

(4997 more lines)
```

### 评价

使用 AUC 最后评判标准。

下面简单介绍AUC，介绍AUC之前先介绍ROC：

![](/assets/p8947349.jpg)

正如我们在这个ROC曲线的示例图中看到的那样，ROC曲线的横坐标为false positive rate（FPR），纵坐标为true positive rate（TPR）。下图中详细说明了FPR和TPR是如何定义的。

![](/assets/p8947350.jpg)

接下来我们考虑ROC曲线图中的四个点和一条线。第一个点，\(0,1\)，即FPR=0, TPR=1，这意味着FN（false negative）=0，并且FP（false positive）=0。这是一个完美的分类器，它将所有的样本都正确分类。第二个点，\(1,0\)，即FPR=1，TPR=0，类似地分析可以发现这是一个最糟糕的分类器，因为它成功避开了所有的正确答案。第三个点，\(0,0\)，即FPR=TPR=0，即FP（false positive）=TP（true positive）=0，可以发现该分类器预测所有的样本都为负样本（negative）。类似的，第四个点（1,1），分类器实际上预测所有的样本都为正样本。经过以上的分析，我们可以断言，ROC曲线越接近左上角，该分类器的性能越好。

**AUC（Area Under Curve）**被定义为ROC曲线下的面积，显然这个面积的数值不会大于1。又由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围在0.5和1之间。使用AUC值作为评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC更大的分类器效果更好。

既然已经这么多评价标准，为什么还要使用ROC和AUC呢？因为ROC曲线有个很好的特性：当测试集中的正负样本的分布变化的时候，ROC曲线能够保持不变。在实际的数据集中经常会出现类不平衡（class imbalance）现象，即负样本比正样本多很多（或者相反），而且测试数据中的正负样本的分布也可能随着时间变化。这时使用AUC评价指标就会更加准确。

### 代码与数据

* labeledTrainData: [https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/labeledTrainData.tsv](https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/labeledTrainData.tsv)
* testData: [https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/testData.tsv](https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/testData.tsv)
* unlabeledTrainData: [https://www.kaggle.com/c/word2vec-nlp-tutorial/download/unlabeledTrainData.tsv.zip](https://www.kaggle.com/c/word2vec-nlp-tutorial/download/unlabeledTrainData.tsv.zip)
* sampleSubmission: [https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/sampleSubmission.csv](https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/sampleSubmission.csv)
* correct\_submission: [https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/correct\_submission.csv](https://github.com/GilYexiao/DataSet/raw/master/BagOfWords/correct_submission.csv)

### 完整代码

* MultinomialNB version: [https://www.kaggle.com/ayanmaity/bag-of-words-meets-popcorns](https://www.kaggle.com/jiuzhang/jiuzhang-knn-sk-learn)
* 官方教程：[https://www.kaggle.com/c/word2vec-nlp-tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial)

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



