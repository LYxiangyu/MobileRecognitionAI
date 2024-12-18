import os
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout, QMessageBox, QSizePolicy
from qfluentwidgets import LineEdit, PrimaryPushButton, CheckBox, BodyLabel, setThemeColor, isDarkTheme
from PyQt5.QtGui import QPixmap, QIcon


class LoginWindow(QWidget):
    """ 登录界面 """

    loginSuccess = pyqtSignal(str)  # 登录成功信号，传递用户名

    def __init__(self, parent=None):
        super().__init__(parent)
        # 设置对象名称
        self.setObjectName("login_interface")  # 为 LoginWindow 设置唯一的对象名称
        # 创建主布局 (水平布局)
        self.main_layout = QHBoxLayout(self)
        self.setLayout(self.main_layout)

        # 确保布局可以扩展
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 创建左侧图片部分
        self.createImageSection()

        # 创建右侧表单部分
        self.createFormSection()

        # 设置窗口效果，移除背景样式
        if isDarkTheme():
            self.setStyleSheet("background-color: transparent;")  # 移除背景样式
        else:
            # 如果不是暗黑模式，则设置白色背景
            self.setStyleSheet("background-color: white;")

        # 连接按钮点击事件
        self.pushButton.clicked.connect(self.onLoginClicked)

    def createImageSection(self):
        """ 创建左侧图片部分 """
        image_layout = QVBoxLayout()
        image_layout.setAlignment(Qt.AlignCenter)

        # 获取当前文件所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, 'background.jpg')

        # 打印图片路径以进行调试
        print("Image path:", image_path)

        # 加载并设置图片
        self.image_label = QLabel(self)
        pixmap = QPixmap(image_path)

        # 检查图片是否成功加载
        if pixmap.isNull():
            print("Failed to load image:", image_path)
        else:
            print("Image loaded successfully")

        self.image_label.setPixmap(pixmap.scaled(1000, 650, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        image_layout.addWidget(self.image_label)

        # 将图片布局添加到主布局中
        self.main_layout.addLayout(image_layout)

    def createFormSection(self):
        """ 创建并布置表单控件 """
        form_layout = QVBoxLayout()
        form_layout.setAlignment(Qt.AlignCenter)

        # 创建 Logo 图片部分
        logo_layout = QVBoxLayout()
        logo_layout.setAlignment(Qt.AlignCenter)  # 使 Logo 居中显示

        # 获取当前文件所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_image_path = os.path.join(current_dir, 'kunkun.png')  # 使用与左侧相同的图片路径

        # 打印 Logo 图片路径以进行调试
        print("Logo image path:", logo_image_path)

        # 加载并设置 Logo 图片
        self.logo_label = QLabel(self)
        logo_pixmap = QPixmap(logo_image_path)

        # 检查 Logo 图片是否成功加载
        if logo_pixmap.isNull():
            print("Failed to load logo image:", logo_image_path)
        else:
            print("Logo image loaded successfully")

        self.logo_label.setPixmap(logo_pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo_layout.addWidget(self.logo_label)

        # 将 Logo 布局添加到表单布局的顶部
        form_layout.addLayout(logo_layout)

        # 添加欢迎标题
        welcome_label = BodyLabel("欢迎回来！")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        form_layout.addWidget(welcome_label)

        # 添加一些间距
        form_layout.addSpacing(30)

        # 登录按钮
        self.pushButton = PrimaryPushButton("登录")
        form_layout.addWidget(self.pushButton)

        # 将表单布局添加到主布局中
        self.main_layout.addLayout(form_layout)

    def onLoginClicked(self):
        """ 登录按钮点击事件 """
        # 这里可以根据需要实现登录逻辑
        self.showSuccess("默认用户")

    def showSuccess(self, username):
        """ 显示登录成功提示并发出信号 """

        self.loginSuccess.emit(username)

    def showError(self, message):
        """ 显示错误提示 """
        QMessageBox.critical(self, "登录失败", message)