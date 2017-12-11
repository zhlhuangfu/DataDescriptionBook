# Otto Group 数据集

## 给定Otto Group一些商品的多项特征信息，需要学习一个模型来判断一个商品所属的类别

Link: [https://www.kaggle.com/c/otto-group-product-classification-challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge)

### 题目描述

Otto Group是世界上最大的电子商务公司之一，在全世界范围内，它每天会卖出数百万件商品。每件商品所属的类别（Class\_1～ Class\_9）对于这家公司的货物供给和需求分析是非常重要的信息。现给定一些商品的多项特征信息，你需要学习一个模型来判断一个商品所属的类别。

### 先修技能

* 懂得基本的机器学习分类模型的原理和使用，如SVM，Decision Tree等。
* 懂得集成学习的相关算法，如Random Forest，Adaboost等。
* 懂得神经网络的使用，如MLP等。

### 输入格式

数据文件train.csv和test.csv包含多个商品的信息。每个商品有如下信息：

1. id: 每个商品的唯一id
2. feat\_1 ~ feat\_93: 每个商品的93个特征
3. target: 商品的真正类别\(仅train.csv中包含此信息\)

训练数据集\(train.csv\)包含95列，分别对应上述信息。测试数据集\(test.csv\)包含94列，不包含target信息。

### 输出格式

您需要提交一个csv文件，文件应采用以下格式：对于测试集中的每个商品，输出一行，其中包含商品的id和您预测的其分别属于Class\_1 ~ Class\_9的概率（要求概率和为1）。 您的提交文件将如下所示：

```
id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9
1,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0
2,0.0,0.2,0.3,0.3,0.0,0.0,0.1,0.1,0.0
...
etc.
```

### 评价

使用 multi-class logarithmic loss 作为最后评判标准，公式如下：


$$
logloss = -\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^My_{ij}log(p_{ij})
$$


其中$$N$$代表测试数据集中的商品数量，$$log$$使用自然对数，$$y_{ij}$$表示商品$$i$$是否属于$$class_j$$，如果是则$$y_{ij}=1$$，否则为$$0$$。$$p_{ij}$$代表你预测商品$$i$$属于$$class_j$$的概率。

### 代码与数据

* train : [https://github.com/GaoChengliang/MLData/blob/master/Otto/train.csv](https://github.com/GaoChengliang/MLData/blob/master/Otto/train.csv)
* test : [https://github.com/GaoChengliang/MLData/blob/master/Otto/test.csv](https://github.com/GaoChengliang/MLData/blob/master/Otto/test.csv)
* correct\_submission : [https://github.com/GaoChengliang/MLData/blob/master/Otto/otto\_correct\_submission.csv](https://github.com/GaoChengliang/MLData/blob/master/Otto/otto_correct_submission.csv)

### 完整代码

* [https://gist.github.com/chrisdubois/6b93a8028f4dc40cab49](https://gist.github.com/chrisdubois/6b93a8028f4dc40cab49)

### 参考分析

* [https://www.kaggle.com/tqchen/understanding-xgboost-model-on-otto-data](https://www.kaggle.com/tqchen/understanding-xgboost-model-on-otto-data)

### 测评配置环境

python

```
pip install -U scikit-learn
pip install -U numpy
pip install -U pandas
```

测评代码

```py
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

test = pd.read_csv("./otto_correct_submission.csv").sort_values('id')
pred = pd.read_csv("./prediction_test.csv").sort_values('id')

test.drop('id', axis=1, inplace=True)
pred.drop('id', axis=1, inplace=True)

loss = log_loss(np.array(test).argmax(axis=1), np.array(pred))
```



