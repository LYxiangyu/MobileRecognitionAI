# 基于Pytorch的简单手机型号识别
手机型号识别系统，可以选择图片并识别。
这只是个简单的模型，应对实训课的要求，上传给学弟学妹们提供方便。
当然，将数据源一换，也可以改造为其他识别。
## 模型：
- YOLO11 [官网链接](https://docs.ultralytics.com/zh/models/yolo11/#usage-examples)
- ResNet152
## 界面：
使用pyqt，组件使用[QFluentWidgets](https://qfluentwidgets.com/)，界面精美
## 数据集
感谢[基于卷积神经网络的手机型号识别系统](https://github.com/haotian02/Mobile-phone-model-recognition-system-based-on-convolutional-neural-network)提供的数据源
使用RoboFlow完成了对数据集的划分。
## 使用
### 环境
请提前安好如下等：
- pytorch
- pyqt
- ultralytics
- PyQt-Fluent-Widgets
- numpy
- openCV
可能并不全面，请根据代码安装。
### 启动
运行loginview.py启动，如要训练，ResNet训练在train3.py，YOLO训练可通过UI界面启动，数据集请提前放在dataset。数据集格式如下
```bash
dataset/
    train/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img3.jpg
            img4.jpg
    val/
    test/
```
