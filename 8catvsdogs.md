# Cats Vs Dogs

## 给出一张猫或狗的图片，识别出这是猫还是狗？

Link: [https://www.kaggle.com/c/dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats) \(对原题数据改动较大\)

### 题目描述

编写一个算法来分类图像是否包含狗或猫。

Web服务为了进行保护，防止一些计算机进行恶意的访问或爬取，会设立一些验证方法，有些识别问题对于人们来说容易解决，但是对于计算机则很困难。这样的方法称为CAPTCHA（完全自动公开的图灵测试）或HIP（人类交互证明）。 HIP有很多用处，例如减少垃圾邮件，防止暴力破解网站密码。

Asirra（用于限制访问的动物图像识别）是一个HIP，询问用户识别猫和狗的照片。这对于计算机很困难，但研究表明，人们可以快速准确地完成任务。以下是Asirra的一个例子：

Asirra与世界上最大的寻找无家可归宠物家园的网站Petfinder.com合作，向微软研究院提供了超过三百万张猫和狗的图像，由美国各地成千上万的动物收容所手动分类。 我们很幸运能够提供这些数据的一个子集，用于学习和研究。

对于进行入侵的计算机，随机猜测是最简单的攻击方法，但图像识别可以让攻击者做出更好的猜测。图片之间（不同的的背景，角度，姿势，亮度等）存在着巨大的差异，难以进行准确的自动分类。在多年前进行的非正式调查中，计算机视觉科学家认为，如果没有现有技术的重大进展，精度高于60％是十分困难的。作为参考，60％分类器将12幅图像HIP的猜测概率从1/4096提高到1/459。

而目前的文献表明机器分类器在这个任务上的准确度可以达到80％以上。你能在猫狗之间分辨出他们的差异吗？

### 先修技能

* 掌握初步的图像处理能力，如转化灰度图像，彩色图像的表示。
* 掌握卷积神经网络及相关的技巧，如SoftMax、ReLu等， 或者svm等较强的分类器。

### 输入格式

* train.rar包含了20000个猫和狗的jpg图片这些图片大小不尽相同，在这些图片上进行训练。\(1=狗，0=猫\)的标签其中id即为'.jpg'前面的文件编号

* test.rar是用来预测的测试集数据。 建议不要手动修改test的预测Label。

### 输出格式

您的提交csv文件应包含行名，并采用以下格式：对于测试集中的每张图片，输出一行，其中包含图片id和对应预测的结果（**1=狗，0=猫**）。如下所示：

```
id,label
1,1
2,1
3,0 

(etc...)
```

### 评价

采用Accuracy评估

### 代码与数据

* **train.rar是训练集**、**test.rar是测试集**、**sampleSubmission.csv是提交示例**、**correctSubmission.csv是正确答案**：

* 这里**数据自己切分了下，**

* **由于过大放Github不太方便，直接放百度云盘**： 链接：[https://pan.baidu.com/s/1i44wmpv](https://pan.baidu.com/s/1i44wmpv) 密码：9bpu

### 完整代码

* \(CNN\_relu\_adam\_softmax\) [https://www.kaggle.com/nkummer/cat-v-dog](https://www.kaggle.com/nkummer/cat-v-dog)

### 测评配置环境

python

```
pip install -U numpy
pip install pandas
pip install -U scikit-learn
```

测评代码

```py
from sklearn.metrics import accuracy_score
import pandas as pd
y_test = pd.read_csv(data_dir + "correctSubmission.csv") # 正确答案
y_pred = pd.read_csv(data_dir + "predictionTest.csv") # 用户预测的答案
auc = accuracy_score(y_test['sentiment'],y_pred['sentiment'])
```



