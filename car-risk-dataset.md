# Car Risk数据集

## 给定汽车的各项指标，设计算法对汽车的投保风险进行打分

### 题目描述

某保险公司销售一种汽车保险，需要对汽车状态进行评估。现在你需要学习一个模型，可以根据汽车的各项指标对汽车的投保风险进行打分。投保风险是从0到70的正整数，数值越大代表风险越高。

### 先修技能

* 懂得基本的机器学习回归模型的原理和使用，如SVR，CART等。
* 懂得集成学习的相关算法，如Random Forest，Adaboost等。

### 输入格式

数据文件train.csv和test.csv包含多辆汽车的信息。每辆汽车有如下信息：

1. Id: 每个汽车的唯一id
2. Score: 汽车的风险数值\(整数，仅train.csv中包含此信息\)
3. Col\_1 ~ Col\_32: 每个汽车的32个特征，其中有数值型，有类别型\(用字母表示）

训练数据集\(train.csv\)包含34列，分别对应上述信息。测试数据集\(test.csv\)包含33列，不包含Score信息。

### 输出格式

您需要提交一个csv文件，文件应采用以下格式：对于测试集中的每辆汽车，输出一行，其中包含汽车的Id和您预测的风险值。 您的提交文件将如下所示：

```
Id,Score
1,5
2,6
...
etc.
```

### 评价

使用RMSE \(Root Mean Square Error\)作为评价指标，公式如下：

<img src="http://www.forkosh.com/mathtex.cgi? RMSE=\sqrt{\frac{\sum_{i=1}^N(y_i-\hat{y_i})^2}{N}}">

其中<img src="http://www.forkosh.com/mathtex.cgi? N">代表测试数据集中汽车的数量，<img src="http://www.forkosh.com/mathtex.cgi? y_{i}">代表其真实的风险值，<img src="http://www.forkosh.com/mathtex.cgi? \hat{y_i}">代表你预测的风险值。

### 代码与数据

* train : [https://github.com/GaoChengliang/MLData/blob/master/Car/train.csv](https://github.com/GaoChengliang/MLData/blob/master/Car/train.csv)
* test : [https://github.com/GaoChengliang/MLData/blob/master/Car/test.csv](https://github.com/GaoChengliang/MLData/blob/master/Car/test.csv)
* correct\_submission : [https://github.com/GaoChengliang/MLData/blob/master/Car/car\_risk\_correct\_submission.csv](https://github.com/GaoChengliang/MLData/blob/master/Car/car_risk_correct_submission.csv)

### 完整代码

* [https://github.com/GaoChengliang/MLData/tree/master/Car/Code](https://github.com/GaoChengliang/MLData/tree/master/Car/Code)

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
from sklearn.metrics import mean_squared_error

test = pd.read_csv("./car_risk_correct_submission.csv").sort_values('Id')
pred = pd.read_csv("./prediction_test.csv").sort_values('Id')

rmse =np.sqrt(mean_squared_error(test['Score'], pred['Score']))
```



