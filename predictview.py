import sys
import os
import time
import subprocess
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QFrame, \
    QFileDialog, QSizePolicy, QScrollArea
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from qfluentwidgets import (
    PrimaryPushButton, CardWidget, FluentIcon, setTheme, Theme, InfoBar, InfoBarPosition, ScrollArea, SimpleCardWidget
)
from ultralytics import YOLO

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

    def update_result(self, result):
        """ 更新识别结果标签 """
        self.result_label.setText(f"识别结果: {result}")

    def update_error(self, error):
        """ 更新错误信息标签 """
        self.result_label.setText(error)

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






