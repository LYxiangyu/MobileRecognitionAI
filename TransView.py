import sys
from qfluentwidgets import FluentIcon as FIF, NavigationAvatarWidget
from PyQt5.QtCore import Qt, QPropertyAnimation, QSize, QRect, QPoint, QUrl, pyqtSignal
from PyQt5.QtGui import QIcon, QColor, QFont, QPainter
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QGraphicsOpacityEffect, QHBoxLayout
from qfluentwidgets import CardWidget, MSFluentWindow, FluentIcon, NavigationItemPosition, ScrollArea, isDarkTheme, \
    TransparentToolButton, HorizontalFlipView, BodyLabel, PillPushButton, setFont, SimpleCardWidget, ImageLabel, \
    TitleLabel, PrimaryPushButton, HyperlinkLabel, VerticalSeparator, HeaderCardWidget, GroupHeaderCardWidget, \
    PushButton, ComboBox, SearchLineEdit
from qfluentwidgets.components.widgets.acrylic_label import AcrylicBrush

from ValView import ValView
from predictview import PredictView
from train import startTraining, TrainingThread


class AppInfoCard(SimpleCardWidget):
    """ App information card """
    start_training_signal = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.iconLabel = ImageLabel("UI/resource/logo/训练.jpg", self)
        self.iconLabel.setBorderRadius(8, 8, 8, 8)
        self.iconLabel.scaledToWidth(120)

        self.nameLabel = TitleLabel('开始训练吧！', self)
        self.startButton = PrimaryPushButton('Start', self)
        self.startButton.clicked.connect(self.on_start_button_clicked)  # 连接按钮点击事件
        self.companyLabel = HyperlinkLabel(
            QUrl('https://github.com/LYxiangyu/MobileRecognitionAI'), 'Xiangyu2233 && Wugaga_233', self)
        self.startButton.setFixedWidth(160)
        self.separator = VerticalSeparator(self)

        self.descriptionLabel = BodyLabel(
            '这里是数据训练。在这里，你可以启动并开始训练，在下方的模块，你可以对训练的方式进行简单的修改。训练的图片如下方图片所示，你可以在/dataset文件夹中，添加数据的训练和验证数据集', self)
        self.descriptionLabel.setWordWrap(True)

        self.tagButton = PillPushButton('数据训练', self)
        self.tagButton.setCheckable(False)
        setFont(self.tagButton, 12)
        self.tagButton.setFixedSize(80, 32)

        self.hBoxLayout = QHBoxLayout(self)
        self.vBoxLayout = QVBoxLayout()
        self.topLayout = QHBoxLayout()
        self.statisticsLayout = QHBoxLayout()
        self.buttonLayout = QHBoxLayout()

        self.initLayout()
        self.setBorderRadius(8)

    def on_start_button_clicked(self):
        self.start_training_signal.emit()  # 点击按钮时发射信号

    def initLayout(self):
        self.hBoxLayout.setSpacing(30)
        self.hBoxLayout.setContentsMargins(34, 24, 24, 24)
        self.hBoxLayout.addWidget(self.iconLabel)
        self.hBoxLayout.addLayout(self.vBoxLayout)

        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.setSpacing(0)

        # name label and install button
        self.vBoxLayout.addLayout(self.topLayout)
        self.topLayout.setContentsMargins(0, 0, 0, 0)
        self.topLayout.addWidget(self.nameLabel)
        self.topLayout.addWidget(self.startButton, 0, Qt.AlignRight)

        # company label
        self.vBoxLayout.addSpacing(3)
        self.vBoxLayout.addWidget(self.companyLabel)

        # statistics widgets
        self.vBoxLayout.addSpacing(20)
        self.vBoxLayout.addLayout(self.statisticsLayout)
        self.statisticsLayout.setContentsMargins(0, 0, 0, 0)
        self.statisticsLayout.setSpacing(10)
        self.statisticsLayout.addWidget(self.separator)
        self.statisticsLayout.setAlignment(Qt.AlignLeft)

        # description label
        self.vBoxLayout.addSpacing(20)
        self.vBoxLayout.addWidget(self.descriptionLabel)

        # button
        self.vBoxLayout.addSpacing(12)
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addLayout(self.buttonLayout)
        self.buttonLayout.addWidget(self.tagButton, 0, Qt.AlignLeft)



class LightBox(QWidget):
    """ Light box """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        if isDarkTheme():
            tintColor = QColor(32, 32, 32, 200)
        else:
            tintColor = QColor(255, 255, 255, 160)

        self.acrylicBrush = AcrylicBrush(self, 30, tintColor, QColor(0, 0, 0, 0))

        self.opacityEffect = QGraphicsOpacityEffect(self)
        self.opacityAni = QPropertyAnimation(self.opacityEffect, b"opacity", self)
        self.opacityEffect.setOpacity(1)
        self.setGraphicsEffect(self.opacityEffect)

        self.vBoxLayout = QVBoxLayout(self)
        self.closeButton = TransparentToolButton(FluentIcon.CLOSE, self)
        self.flipView = HorizontalFlipView(self)
        self.nameLabel = BodyLabel('样例图片 1', self)
        self.pageNumButton = PillPushButton('1 / 4', self)

        self.pageNumButton.setCheckable(False)
        self.pageNumButton.setFixedSize(80, 32)
        setFont(self.nameLabel, 16, QFont.DemiBold)

        self.closeButton.setFixedSize(32, 32)
        self.closeButton.setIconSize(QSize(14, 14))
        self.closeButton.clicked.connect(self.fadeOut)

        self.vBoxLayout.setContentsMargins(26, 28, 26, 28)
        self.vBoxLayout.addWidget(self.closeButton, 0, Qt.AlignRight | Qt.AlignTop)
        self.vBoxLayout.addWidget(self.flipView, 1)
        self.vBoxLayout.addWidget(self.nameLabel, 0, Qt.AlignHCenter)
        self.vBoxLayout.addSpacing(10)
        self.vBoxLayout.addWidget(self.pageNumButton, 0, Qt.AlignHCenter)

        self.flipView.addImages([
            'UI/resource/样例图片/Huawei.png', 'UI/resource/样例图片/OPPO.png',
            'UI/resource/样例图片/Vivo.jpeg', 'UI/resource/样例图片/Xiaomi.jpeg',
        ])
        self.flipView.currentIndexChanged.connect(self.setCurrentIndex)

    def setCurrentIndex(self, index: int):
        self.nameLabel.setText(f'样例图片 {index + 1}')
        self.pageNumButton.setText(f'{index + 1} / {self.flipView.count()}')
        self.flipView.setCurrentIndex(index)

    def paintEvent(self, e):
        if self.acrylicBrush.isAvailable():
            return self.acrylicBrush.paint()

        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        if isDarkTheme():
            painter.setBrush(QColor(32, 32, 32))
        else:
            painter.setBrush(QColor(255, 255, 255))

        painter.drawRect(self.rect())

    def resizeEvent(self, e):
        w = self.width() - 52
        self.flipView.setItemSize(QSize(w, w * 9 // 16))

    def fadeIn(self):
        rect = QRect(self.mapToGlobal(QPoint()), self.size())
        self.acrylicBrush.grabImage(rect)

        self.opacityAni.setStartValue(0)
        self.opacityAni.setEndValue(1)
        self.opacityAni.setDuration(150)
        self.opacityAni.start()
        self.show()

    def fadeOut(self):
        self.opacityAni.setStartValue(1)
        self.opacityAni.setEndValue(0)
        self.opacityAni.setDuration(150)
        self.opacityAni.finished.connect(self._onAniFinished)
        self.opacityAni.start()

    def _onAniFinished(self):
        self.opacityAni.finished.disconnect()
        self.hide()

class GalleryCard(HeaderCardWidget):
    """ Gallery card """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('样例图片')
        self.setBorderRadius(8)

        self.flipView = HorizontalFlipView(self)
        self.expandButton = TransparentToolButton(
            FluentIcon.CHEVRON_RIGHT_MED, self)

        self.expandButton.setFixedSize(32, 32)
        self.expandButton.setIconSize(QSize(12, 12))

        self.flipView.addImages([
            'UI/resource/样例图片/Huawei.png', 'UI/resource/样例图片/OPPO.png',
            'UI/resource/样例图片/Vivo.jpeg', 'UI/resource/样例图片/Xiaomi.jpeg',
        ])
        self.flipView.setBorderRadius(8)
        self.flipView.setSpacing(10)

        self.headerLayout.addWidget(self.expandButton, 0, Qt.AlignRight)
        self.viewLayout.addWidget(self.flipView)

class SettinsCard(GroupHeaderCardWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("基本设置")
        self.setBorderRadius(8)

        self.chooseButton = PushButton("选择")
        self.comboBox = ComboBox()
        self.lineEdit = SearchLineEdit()

        self.chooseButton.setFixedWidth(120)
        self.lineEdit.setFixedWidth(320)
        self.comboBox.setFixedWidth(320)
        self.lineEdit.setPlaceholderText("默认100")
        self.addGroup("resource/Python.svg", "训练轮数", "请填写训练轮数", self.lineEdit)


class AppInterface(ScrollArea):
    start_training_signal = pyqtSignal()  # 定义信号

    def __init__(self, parent=None):
        super().__init__(parent)

        self.view = QWidget(self)

        self.vBoxLayout = QVBoxLayout(self.view)
        self.appCard = AppInfoCard(self)
        self.galleryCard = GalleryCard(self)
        self.settingCard = SettinsCard(self)
        self.appCard.start_training_signal.connect(self.on_start_training_signal)  # 接收信号
        self.lightBox = LightBox(self)
        self.lightBox.hide()
        self.galleryCard.flipView.itemClicked.connect(self.showLightBox)

        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setObjectName("appInterface")

        self.vBoxLayout.setSpacing(10)
        self.vBoxLayout.setContentsMargins(0, 0, 10, 30)
        self.vBoxLayout.addWidget(self.appCard, 0, Qt.AlignTop)
        self.vBoxLayout.addWidget(self.galleryCard, 0, Qt.AlignTop)
        self.vBoxLayout.addWidget(self.settingCard, 0, Qt.AlignTop)

        self.enableTransparentBackground()

    def showLightBox(self):
        index = self.galleryCard.flipView.currentIndex()
        self.lightBox.setCurrentIndex(index)
        self.lightBox.fadeIn()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.lightBox.resize(self.size())

    def on_start_training_signal(self):
        """处理 AppInfoCard 中点击按钮后发出的信号"""
        print("AppInterface received the signal.")
        self.start_training_signal.emit()  # 将信号转发给父组件

class Demo3(MSFluentWindow):
    start_training_signal = pyqtSignal()  # 定义信号
    def __init__(self, parent=None):
        super().__init__(parent)

        self.appInterface = AppInterface(self)
        self.valView = ValView(self)
        self.predictview = PredictView(self)
        # 初始化训练线程
        self.training_thread = TrainingThread()
        self.training_thread.training_started.connect(self.on_training_started)
        self.training_thread.training_finished.connect(self.on_training_finished)

        # 连接按钮点击信号到启动训练
        self.appInterface.start_training_signal.connect(self.on_start_training)
        # add sub interfaces
        # self.addSubInterface(self.appInterface, FluentIcon.LIBRARY, "训练", FluentIcon.LIBRARY_FILL, isTransparent=True)
        # self.navigationInterface.addItem("editInterface", FluentIcon.EDIT, "编辑", selectable=False)
        #
        # self.navigationInterface.addItem(
        #     "settingInterface", FluentIcon.SETTING, "设置", position=NavigationItemPosition.BOTTOM, selectable=False)
        self.initNavigation()
        self.resize(880, 760)
        self.setWindowTitle('手机型号识别')
        self.setWindowIcon(QIcon('UI/resource/logo/icon.png'))

        self.titleBar.raise_()

    def on_start_training(self):
        print("Training started...")
        self.training_thread.start()  # 启动训练线程

    def on_training_started(self):
        print("Training has started.")

    def on_training_finished(self):
        print("Training has finished.")
        # 在这里可以更新界面，如禁用按钮或显示结果
    def initNavigation(self):
        self.addSubInterface(self.appInterface, FluentIcon.ROBOT, "训练",  isTransparent=True)
        self.addSubInterface(self.valView, FluentIcon.PENCIL_INK, "验证",  isTransparent=True)
        self.addSubInterface(self.predictview, FluentIcon.SEARCH, "识别",  isTransparent=True)
        # self.addSubInterface( FIF.SETTING, 'Settings', NavigationItemPosition.BOTTOM)
        # self.stackWidget.setCurrentIndex(1)

if __name__ == '__main__':
    # enable dpi scale
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # setTheme(Theme.DARK)

    app = QApplication(sys.argv)
    w3 = Demo3()
    w3.start_training_signal.connect(startTraining)
    w3.show()
    app.exec_()