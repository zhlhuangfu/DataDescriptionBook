# Criteo Display Advertising Challenge 

## 依据Criteo提供的广告展示数据，预测用户是否会点击广告。

Link: [https://www.kaggle.com/c/criteo-display-ad-challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)

### 题目描述

Criteo是一家第三方广告展示公司，与世界上超过4000家电子商务公司有合作关系。本题我们使用Criteo所共享的一周广告展示数据，数据中提炼了13个连续特征和26个离散特征和用户是否点击了该页面广告的标签。请你训练出合适的模型，预测用户在不同的特征下是否会点击广告。

### 先修技能

* GBDT等相关知识
* Logarithmic Loss 
* Logic Regression

### 输入格式
数据文件train.csv包含了1599条访问特征。
提供了一系列的用户访问网页和点击广告的记录，l1～l13为一些计数特征，c1～c26为一些类别特征。Label表示用户是否点击广告，0为未点击，1为点击。

```
Id,Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26
10000743,1,1.0,0,1.0,,227.0,1.0,173.0,18.0,50.0,1.0,7.0,1.0,,75ac2fe6,1cfdf714,713fbe7c,aa65a61e,25c83c98,3bf701e7,7195046d,0b153874,a73ee510,9e5006cd,4d8549da,a48afad2,51b97b8f,b28479f6,d345b1a0,3fa658c5,3486227d,e88ffc9d,c393dc22,b1252a9d,57c90cd9,,bcdee96c,4d19a3eb,cb079c2d,456c12a0
10000159,1,4.0,1,1.0,2.0,27.0,2.0,4.0,2.0,2.0,1.0,1.0,,2.0,05db9164,6c9c9cf3,2730ec9c,5400db8b,25c83c98,7e0ccccf,8a6600b0,813607cc,a73ee510,e4b08fda,4ab39743,be45b877,ab8a1a53,07d13a8f,06969a20,9bc7fff5,07c540c4,92555263,,,242bb710,,3a171ecb,72c78f11,,
etc...
```

### 输出格式
根据测试集给出的用户访问记录，预测出用户点击某个广告的概率，第一列为记录Id，第二列为点击概率。输出格式如下所示：

```
Id,Predicted
60000000,0.384
63895816,0.5919
759281658,0.1934
895936184,0.9572
etc...
```

### 评价

使用Logarithmic Loss (https://www.kaggle.com/wiki/LogarithmicLoss) 作为最后评判标准,公式如下：
<img src="http://www.forkosh.com/mathtex.cgi? logloss=-\frac{1}{n}\sum_{i=1}^N\sum_{j=1}^My_{i,j}\log(p_{i,j})">
Logarithmic Loss可以用来表征预测值和标准值的相关性。



### 代码与数据

* train：[https://github.com/wfnuser/my-kaggle-dataset/blob/master/ctr/train.csv](https://github.com/wfnuser/my-kaggle-dataset/blob/master/ctr/train.csv)
* test：[https://github.com/wfnuser/my-kaggle-dataset/blob/master/ctr/test.csv](https://github.com/wfnuser/my-kaggle-dataset/blob/master/ctr/test.csv)
* correct_submission: [https://github.com/wfnuser/my-kaggle-dataset/blob/master/ctr/correct_submission.csv](https://github.com/wfnuser/my-kaggle-dataset/blob/master/ctr/correct_submission.csv)

### 完整代码

https://github.com/guestwalk/kaggle-2014-criteo


### 测评配置环境

python

```
pip install -U numpy
pip install pandas
pip install -U scikit-learn
```

测评代码

```py
from sklearn.metrics import log_loss
y_test = pd.read_csv(data_dir + "correct_submission.csv")
y_pred = pd.read_csv(data_dir + "prediction_test.csv")
accuracy = log_loss(y_test,y_pred)
```