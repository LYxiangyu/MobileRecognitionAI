import logging
import sys
import os
import time
import subprocess
import numpy as np
import torch
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QFrame, \
    QFileDialog, QSizePolicy, QScrollArea
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from qfluentwidgets import (
    PrimaryPushButton, CardWidget, FluentIcon, setTheme, Theme, InfoBar, InfoBarPosition, ScrollArea, SimpleCardWidget
)
from torchvision import transforms, models
from ultralytics import YOLO

from predict2 import load_classes
from train2 import SimpleCNN

# 加载 YOLO 模型为全局对象
YOLO_MODEL = YOLO("./runs/classify/train/weights/best.pt")

class PredictThread(QThread):
    result_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            results = YOLO_MODEL(self.image_path, show=False)
            names_dict = results[0].names
            probs = results[0].probs.data.tolist()
            predicted_class = names_dict[np.argmax(probs)]
            self.result_signal.emit(predicted_class)
        except Exception as e:
            self.error_signal.emit(f"错误: {str(e)}")


class UpPhoneCard(SimpleCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vBoxLayout = QVBoxLayout(self)
        self.setContentsMargins(10, 10, 10, 10)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: transparent; border: 2px dashed gray;")
        self.image_label.setFixedSize(300, 600)
        self.vBoxLayout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.selectButton = PrimaryPushButton('选择图片', self)
        self.selectButton.clicked.connect(self.open_image)  # 连接按钮点击事件
        self.vBoxLayout.addWidget(self.selectButton,alignment=Qt.AlignCenter)
        self.current_image_path = None

    def open_image(self):
        """ 打开文件管理器并显示选择的图片 """
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_path:
            self.display_image(file_path)
            self.current_image_path = file_path

    def display_image(self, file_path):
        """ 在 image_label 中显示选中的图片 """
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

class ResultCard(SimpleCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vBoxLayout = QVBoxLayout(self)
        self.setContentsMargins(10, 10, 10, 10)

        # 结果标签
        self.result_label = QLabel("识别结果", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 16))
        self.vBoxLayout.addWidget(self.result_label)

        # 提交按钮
        self.submit_button = PrimaryPushButton("提交图片", self)
        self.submit_button.clicked.connect(self.run_model)  # 连接按钮点击事件
        self.vBoxLayout.addWidget(self.submit_button, alignment=Qt.AlignCenter)

        self.image_path = None  # 存储图片路径
        self.predict_thread = None  # 预测线程
        self.yolo_result = ""  # 保存 YOLO 结果
        self.handwritten_result = ""  # 保存手写模型结果
        self.handwritten_model = self.load_handwritten_model()

    def set_image_path(self, path):
        if os.path.exists(path):
            self.image_path = path
        else:
            self.result_label.setText("图片路径无效，请重新上传！")

    def run_model(self):
        if not self.image_path:
            self.result_label.setText("请先上传一张图片！")
            return

        self.result_label.setText("正在预测，请稍候...")

        self.predict_thread = PredictThread(self.image_path)
        self.predict_thread.result_signal.connect(self.update_result)
        self.predict_thread.error_signal.connect(self.update_error)
        self.predict_thread.start()
        # 运行手写模型推理
        handwritten_prediction, handwritten_probability = self.predict_handwritten_model(self.image_path)
        if handwritten_prediction:
            self.handwritten_result = f"ResNet-152模型: {handwritten_prediction}"
            self.update_final_result()

    def update_result(self, result):
        """ 更新识别结果标签 """
        self.yolo_result = f"YOLO 结果: {result}"
        self.update_final_result()

    def update_final_result(self):
        """ 更新最终显示的结果，将 YOLO 和手写模型结果一起显示 """
        combined_result = "\n".join(filter(None, [self.yolo_result, self.handwritten_result]))
        self.result_label.setText(combined_result)

    def update_error(self, error):
        """ 更新错误信息标签 """
        self.result_label.setText(error)

    def load_handwritten_model(self):
        """加载手写模型"""
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # 加载类别名称
        classes_file = './runs/classify/train2/classes.txt'
        if not os.path.exists(classes_file):
            logging.error(f"Classes file not found at {classes_file}")
            return None
        self.classes = load_classes(classes_file)
        num_classes = len(self.classes)

        # 加载 ResNet-152 模型
        model = models.resnet152(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        model = model.to(device)

        model_path = './runs/classify/train2/weights/best3.pth'
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at {model_path}")
            return None
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logging.info(f"Loaded ResNet-152 model from {model_path}")
        return model

    def predict_handwritten_model(self, image_path):
        """使用手写模型进行预测"""
        if not self.handwritten_model:
            return None, None

        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 加载图像并预测
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = self.handwritten_model(image)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1)
                prob = probs[0, pred].item()
            return self.classes[pred.item()], prob
        except Exception as e:
            logging.error(f"Handwritten model prediction failed: {str(e)}")
            return None, None

class PredictView(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 主布局
        self.view = QWidget(self)
        self.hBoxLayout = QHBoxLayout(self.view)
        self.hBoxLayout.setContentsMargins(10, 10, 10, 10)
        self.hBoxLayout.setSpacing(10)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setObjectName("predictView")
        # 左侧卡片：上传图片
        self.upload_card = UpPhoneCard(self)
        self.hBoxLayout.addWidget(self.upload_card)
        # 右侧卡片：识别结果
        self.result_card = ResultCard(self)
        self.hBoxLayout.addWidget(self.result_card)
        # 绑定上传与提交功能
        self.upload_card.selectButton.clicked.connect(self.update_result_card)
        self.enableTransparentBackground()

    def update_result_card(self):
        """ 更新 ResultCard 的图片路径 """
        image_path = self.upload_card.current_image_path
        if image_path:
            self.result_card.set_image_path(image_path)






