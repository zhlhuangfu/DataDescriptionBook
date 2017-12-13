# Facial Keypoints Detection

## 你能教会计算机识别人脸的关键部位吗？

Link: [https://www.kaggle.com/c/facial-keypoints-detection](https://www.kaggle.com/c/facial-keypoints-detection)

### 题目描述

从人物头像的96x96像素的灰度照片中找出代表面部器官位置的关键点（keypoints）的坐标，关键点包括左右眼中心等，共15个。
人脸关键点检测是一个非常困难的问题，不同图片的灯光、角度、人脸尺寸都会导致脸部特征的巨大不同。计算机视觉的研究者花了很多年去克服这一领域内的各种困难，现在得到的结果非常令人惊喜，但仍然还有很多问题值得探索。

### 先修技能

* CNN的相关知识。

### 输入格式
train.csv提供了大约5000个人物头像的灰度图片，像素96x96，灰度0-255，图片数据的矩阵被整理成一维向量，并附有每个头像15个关键点（keypoints）的位置坐标（x轴y轴）。数据格式如下：
```
left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y,left_eye_inner_corner_x,left_eye_inner_corner_y,left_eye_outer_corner_x,left_eye_outer_corner_y,right_eye_inner_corner_x,right_eye_inner_corner_y,right_eye_outer_corner_x,right_eye_outer_corner_y,left_eyebrow_inner_end_x,left_eyebrow_inner_end_y,left_eyebrow_outer_end_x,left_eyebrow_outer_end_y,right_eyebrow_inner_end_x,right_eyebrow_inner_end_y,right_eyebrow_outer_end_x,right_eyebrow_outer_end_y,nose_tip_x,nose_tip_y,mouth_left_corner_x,mouth_left_corner_y,mouth_right_corner_x,mouth_right_corner_y,mouth_center_top_lip_x,mouth_center_top_lip_y,mouth_center_bottom_lip_x,mouth_center_bottom_lip_y,Image
66.0335639098,39.0022736842,30.2270075188,36.4216781955,59.582075188,39.6474225564,73.1303458647,39.9699969925,36.3565714286,37.3894015038,23.4528721805,37.3894015038,56.9532631579,29.0336481203,80.2271278195,32.2281383459,40.2276090226,29.0023218045,16.3563789474,29.6474706767,44.4205714286,57.0668030075,61.1953082707,79.9701654135,28.6144962406,77.3889924812,43.3126015038,72.9354586466,43.1307067669,84.4857744361, 96x96 more pixels
etc...
```
### 输出格式

根据测试集给出的头像图片数据，预测出每个人物头像的关键点的位置坐标，第一列为行号，第二列为图像Id，第三列为特征名称，第四列为坐标。输出格式如下所示：

```
RowId,ImageId,FeatureName,Location
1,1,left_eye_center_x,?
2,1,left_eye_center_y,?
3,1,right_eye_center_x,?
4,1,right_eye_center_y,?
etc...
```

### 评价

使用RMSE (Root Mean Square Error)作为评价指标，公式如下：
<img src="http://www.forkosh.com/mathtex.cgi? RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)">
其中代表测试数据集中汽车的数量，代表其真实的风险值，代表你预测的风险值。^2}">

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