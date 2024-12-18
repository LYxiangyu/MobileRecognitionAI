import sys
import os
import time
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QFrame, \
    QFileDialog, QSizePolicy, QScrollArea
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt
from qfluentwidgets import (
    PrimaryPushButton, CardWidget, FluentIcon, setTheme, Theme, InfoBar, InfoBarPosition, ScrollArea, SimpleCardWidget
)

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

    def open_image(self):
        """ 打开文件管理器并显示选择的图片 """
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_path:
            self.display_image(file_path)

    def display_image(self, file_path):
        """ 在 image_label 中显示选中的图片 """
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

class ResultCard(SimpleCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)



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

        self.enableTransparentBackground()






