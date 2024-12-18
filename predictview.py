import sys
import os
import time
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QFrame, \
    QFileDialog, QSizePolicy
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt
from qfluentwidgets import (
    PrimaryPushButton, CardWidget, FluentIcon, setTheme, Theme, InfoBar, InfoBarPosition
)

# 模拟的类别标签（可以根据实际情况修改）
CLASS_LABELS = ['iPhone 12', 'Samsung Galaxy S21', 'Google Pixel 5', 'OnePlus 9', 'Xiaomi Mi 11']

class PhonePredict(QMainWindow):
    def __init__(self):
        super().__init__()
        self.min_width = 800  # 最小宽度
        self.min_height = 800  # 最小高度

        print("初始化UI...")
        self.initUI()
        print("UI初始化完成。")

    def initUI(self):
        self.setWindowTitle("基于PyTorch的手机型号识别系统")
        self.setGeometry(100, 100, self.min_width, self.min_height)
        self.setMinimumSize(self.min_width, self.min_height)

        app_icon = QIcon('path/to/your/project_icon.ico')
        self.setWindowIcon(app_icon)

        # 设置主题为浅色模式
        setTheme(Theme.LIGHT)

        # 设置背景颜色为白色
        self.setStyleSheet("background-color: #F3F6F2;")

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_h_layout = QHBoxLayout()  # 主水平布局
        central_widget.setLayout(main_h_layout)

        # 左侧布局：上传图片框
        left_panel = QVBoxLayout()  # 左侧垂直布局
        main_h_layout.addLayout(left_panel, 7)  # 调整左侧布局占比为70%

        # 使用 CardWidget 包裹上传图片部分
        upload_card = CardWidget(self)
        upload_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)  # 使卡片高度自适应
        card_layout = QVBoxLayout(upload_card)
        card_layout.setContentsMargins(10, 10, 10, 10)  # 设置卡片内边距
        left_panel.addWidget(upload_card)

        # 标题在顶部居中
        title_label = QLabel("上传图片", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))  # 设置字体大小和样式
        title_label.setStyleSheet("background-color: white;color: skyblue;")
        card_layout.addWidget(title_label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: yellow;")
        line.setMaximumHeight(2)  # 减小分割线高度
        card_layout.addWidget(line)

        # 图片展示框，设置最小高度和宽度
        self.image_frame = QLabel(self)
        self.image_frame.setAlignment(Qt.AlignCenter)
        self.image_frame.setStyleSheet("background-color: transparent; border: 2px dashed gray;")
        self.image_frame.setMinimumHeight(600)  # 设置最小高度为400像素
        self.image_frame.setMinimumWidth(400)   # 设置最小宽度为400像素
        card_layout.addWidget(self.image_frame)

        # 添加伸展空间，确保图片框占据更多空间
        card_layout.addStretch()

        # 使用 Fluent-Widgets 的 SecondaryPushButton
        self.upload_button = PrimaryPushButton("选择图片", icon=FluentIcon.PHONE, parent=self)
        self.upload_button.clicked.connect(self.upload_image)
        card_layout.addWidget(self.upload_button)

        # 右侧布局：识别信息
        right_panel = QVBoxLayout()  # 右侧垂直布局
        main_h_layout.addLayout(right_panel, 3)  # 调整右侧布局占比为30%

        # 右侧上方：显示“识别”二字
        recognition_label = QLabel("识别", self)
        recognition_label.setAlignment(Qt.AlignCenter)
        recognition_label.setFont(QFont("Arial", 24, QFont.Bold))

        recognition_label.setStyleSheet("color: skyblue;background-color: white")
        right_panel.addWidget(recognition_label)

        # 右侧中间：水平分割线
        vertical_line = QFrame()
        vertical_line.setFrameShape(QFrame.HLine)
        vertical_line.setFrameShadow(QFrame.Sunken)
        vertical_line.setStyleSheet("color: yellow;")
        vertical_line.setMaximumHeight(2)  # 减小分割线高度
        right_panel.addWidget(vertical_line)

        # 使用 Fluent-Widgets 的 PrimaryPushButton
        self.start_button = PrimaryPushButton("开始", icon=FluentIcon.PLAY, parent=self)
        self.start_button.clicked.connect(self.run_prediction)
        right_panel.addWidget(self.start_button)

        # 识别结果文本框
        self.info_text = QTextEdit(self)
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QFont("Arial", 12))
        self.info_text.setStyleSheet("background-color: transparent; color: red; border: none;")
        right_panel.addWidget(self.info_text)

        self.current_image_path = None
        self.start_time = time.time()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        if self.current_image_path:
            self.display_image(self.current_image_path)
        super().resizeEvent(event)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.xpm *.jpg *.bmp *.gif)")
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.update_info("图片已成功上传。")

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(self.image_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_frame.setPixmap(scaled_pixmap)

    def run_prediction(self):
        if not self.current_image_path:
            self.update_info("请先上传一张图片。")
            return

        # 调用 predict.py 进行图像识别
        try:
            result = subprocess.run(['python', 'predict.py', self.current_image_path], capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            self.update_info(f"预测结果: {output}")
            # 显示成功提示
            InfoBar.success(
                title='成功',
                content='图片识别完成！',
                duration=2000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self
            )
        except subprocess.CalledProcessError as e:
            self.update_info(f"识别失败: {e.stderr.strip()}")
            # 显示错误提示
            InfoBar.error(
                title='错误',
                content=f'图片识别失败: {e.stderr.strip()}',
                duration=2000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self
            )

    def update_info(self, message):
        self.info_text.clear()
        self.info_text.append(message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PhonePredict()
    window.show()
    sys.exit(app.exec_())