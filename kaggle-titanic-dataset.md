# Kaggle Titanic 数据集

Link: [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

### 题目描述

**泰坦尼克号\(RMS Titanic\)**是英国白星航运公司下辖的一艘奥林匹克邮轮，在其处女航行中，因与一座冰山相撞而至沉船。在这次事故中，有约2/3的人丧生。现在给定Titanic上的乘客信息，你需要学习一个模型来判断一名乘客在沉船灾难中能否最终存活下来。

### 先修技能

* 懂得基本的机器学习分类模型的原理和使用，如SVM，Decision Tree等。
* 懂得集成学习的相关算法，如Random Forest，Adaboost等。

### 输入格式

数据文件train.csv和test.csv包含多名乘客的信息。每名乘客有如下信息：

1. PassengerId : 乘客的唯一ID
2. Survived : 乘客最终是否存活\(0 = No, 1 = Yes, 仅train.csv中包含此信息\)
3. Pclass : 乘客的船票的等级\(1 = 1st, 2 = 2nd, 3 = 3rd\)
4. Name : 乘客名字
5. Sex : 乘客性别
6. Age : 乘客年龄\(Year\)
7. Sibsp ：船上兄弟姐妹/配偶的人数
8. Parch : 船上父母/儿女的人数
9. Ticket : 船票号码
10. Fare : 船票价格
11. Cabin : 船舱号
12. Embarked : 出发港口\(C = Cherbourg, Q = Queenstown, S = Southampton\)

训练数据集\(train.csv\)包含12列，分别对应上述信息。测试数据集\(test.csv\)包含11列，不包含Survival信息。

### 输出格式

您需要提交一个csv文件，文件应采用以下格式：对于测试集中的每位乘客，输出一行，其中包含PassengerId和您预测的其是否会存活。 例如，如果您预测第一个乘客存活，第二个乘客不会存活，第三个乘客不会存活，那么您的提交文件将如下所示：

```
PassengerId,Survived
1,1
2,0
3,0 
(415 more lines)
```

### 评价

使用 accuracy 作为最后评判标准。

### 代码与数据

* train : [https://www.kaggle.com/c/titanic/download/train.csv](https://www.kaggle.com/c/titanic/download/train.csv)
* test : [https://www.kaggle.com/c/titanic/download/test.csv](https://www.kaggle.com/c/titanic/download/test.csv)
* correct\_submission : [https://github.com/GaoChengliang/MLData/blob/master/Titanic/titanic\_correct\_submission.csv](https://github.com/GaoChengliang/MLData/blob/master/Titanic/titanic_correct_submission.csv)

### 完整代码

* [https://www.kaggle.com/startupsci/titanic-data-science-solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)

### 测评配置环境

python

```
pip install -U scikit-learn
pip install -U pandas
```

测评代码

```py
import pandas as pd
from sklearn.metrics import accuracy_score
test = pd.read_csv(data_dir + "titanic_correct_submission.csv").sort_values('PassengerId')
pred = pd.read_csv(data_dir + "prediction_test.csv").sort_values('PassengerId')
accuracy = accuracy_score(test['Survived'],pred['Survived'])
```



