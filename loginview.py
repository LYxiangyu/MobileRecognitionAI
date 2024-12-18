import os
import sys

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout, QMessageBox, \
    QSizePolicy, QApplication
from qfluentwidgets import LineEdit, PrimaryPushButton, CheckBox, BodyLabel, setThemeColor, isDarkTheme
from PyQt5.QtGui import QPixmap, QIcon

from TransView import Demo3


class LoginWindow(QWidget):
    """ 登录界面 """

    loginSuccess = pyqtSignal(str)  # 登录成功信号，传递用户名

    def __init__(self,main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window  # 接收主界面引用
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

        self.setWindowTitle('手机型号识别')
        self.setWindowIcon(QIcon(':/qfluentwidgets/images/logo.png'))

        # 连接按钮点击事件
        self.pushButton.clicked.connect(self.onLoginClicked)

    def createImageSection(self):
        """ 创建左侧图片部分 """
        image_layout = QVBoxLayout()
        image_layout.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel(self)
        pixmap = QPixmap('UI/resource/logo/background.jpg')
        self.image_label.setPixmap(pixmap.scaled(1000, 650, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        image_layout.addWidget(self.image_label)

        self.main_layout.addLayout(image_layout)

    def createFormSection(self):
        """ 创建并布置表单控件 """
        form_layout = QVBoxLayout()
        form_layout.setAlignment(Qt.AlignCenter)

        # Logo
        self.logo_label = QLabel(self)
        logo_pixmap = QPixmap('UI/resource/logo/kunkun.png')
        self.logo_label.setPixmap(logo_pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        form_layout.addWidget(self.logo_label)

        # 欢迎标题
        welcome_label = QLabel("欢迎回来！")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        form_layout.addWidget(welcome_label)

        # 登录按钮
        self.pushButton = PrimaryPushButton("登录")
        form_layout.addWidget(self.pushButton)

        self.main_layout.addLayout(form_layout)

    def onLoginClicked(self):
        """ 登录按钮点击事件 """
        self.showSuccess("默认用户")
        self.hide()  # 隐藏登录窗口
        self.main_window.show()  # 显示主界面

    def showSuccess(self, username):
        """ 显示登录成功提示并发出信号 """
        self.loginSuccess.emit(username)

if __name__ == '__main__':
    # enable dpi scale
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # setTheme(Theme.DARK)

    app = QApplication(sys.argv)
    main_window = Demo3()

    # 创建登录窗口并传递主窗口引用
    login_window = LoginWindow(main_window)
    login_window.show()
    app.exec_()