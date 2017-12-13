# Facial Keypoints Detection

## 你能教会计算机识别人脸的关键部位吗？

Link: [https://www.kaggle.com/c/facial-keypoints-detection](https://www.kaggle.com/c/facial-keypoints-detection)

### 题目描述

从人物头像的96x96像素的灰度照片中找出代表面部器官位置的关键点（keypoints）的坐标，关键点包括左右眼中心等，共15个。
人脸关键点检测是一个非常困难的问题，不同图片的灯光、角度、人脸尺寸都会导致脸部特征的巨大不同。计算机视觉的研究者花了很多年去克服这一领域内的各种困难，现在得到的结果非常令人惊喜，但仍然还有很多问题值得探索。

### 先修技能

* CNN的相关知识。

### 输入格式
训练集给了大约5000个人物头像的灰度图片，像素96x96，灰度0-255，图片数据的矩阵被整理成一维向量，并附有每个头像15个关键点（keypoints）的位置坐标（x轴y轴）。

### 输出格式

根据测试集给出的头像图片数据，预测出每个人物头像的关键点的位置坐标，输出格式如下所示：

如下所示：

```
RowId,ImageId,FeatureName,Location
1,1,left_eye_center_x,?
2,1,left_eye_center_y,?
3,1,right_eye_center_x,?
4,1,right_eye_center_y,?
etc...
```

### 评价

使用RMSE作为最后评判标准。
RMSE是方均根偏移(**root-mean-square deviation**)或方均根差(**root-mean-square error**)是一种常用的测量数值之间差异的量度。具体公式如下：
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$


### 代码与数据

* train：[https://www.kaggle.com/c/facial-keypoints-detection/download/training.zip](https://www.kaggle.com/c/facial-keypoints-detection/download/training.zip)
* test：[https://www.kaggle.com/c/facial-keypoints-detection/download/test.zip](https://www.kaggle.com/c/facial-keypoints-detection/download/test.zip)
* samplesubmission: [https://www.kaggle.com/c/facial-keypoints-detection/download/SampleSubmission.csv](https://www.kaggle.com/c/facial-keypoints-detection/download/SampleSubmission.csv)

### 完整代码

* 误差 2.13 ：[http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)



### 测评配置环境

python

```
pip install -U numpy
pip install pandas
pip install -U scikit-learn
```

测评代码

```py
from sklearn.metrics import mean_squared_error
y_test = pd.read_csv(data_dir + "correct_submission.csv")
y_pred = pd.read_csv(data_dir + "prediction_test.csv")
accuracy = mean_squared_error(y_test,y_pred)
```