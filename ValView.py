from PyQt5.QtCore import pyqtSignal, Qt, QUrl
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import SimpleCardWidget, ImageLabel, TitleLabel, PrimaryPushButton, HyperlinkLabel, \
    VerticalSeparator, BodyLabel, PillPushButton, setFont, GroupHeaderCardWidget, PushButton, ComboBox, SearchLineEdit, \
    HeaderCardWidget, ScrollArea


class ValMainInfoCard(SimpleCardWidget):
    """ App information card """
    start_training_signal = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.iconLabel = ImageLabel(":/qfluentwidgets/images/logo.png", self)
        self.iconLabel.setBorderRadius(8, 8, 8, 8)
        self.iconLabel.scaledToWidth(120)

        self.nameLabel = TitleLabel('开始验证吧！', self)
        self.startButton = PrimaryPushButton('Start', self)
        self.companyLabel = HyperlinkLabel(
            QUrl('https://github.com/LYxiangyu/MobileRecognitionAI'), 'Xiangyu2233 && Wugaga_233', self)
        self.startButton.setFixedWidth(160)
        self.separator = VerticalSeparator(self)

        self.descriptionLabel = BodyLabel(
            '这里是数据验证。在这里，你可以启动并开始验证，在下方的模块，你可以对训练的方式进行简单的修改', self)
        self.descriptionLabel.setWordWrap(True)

        self.tagButton = PillPushButton('数据验证', self)
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

class DescriptionCard(HeaderCardWidget):
    """ Description card """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.descriptionLabel = BodyLabel(
            '', self)
        self.vBoxLayout = QVBoxLayout(self.view)
        self.descriptionLabel.setWordWrap(True)
        self.viewLayout.addWidget(self.descriptionLabel)
        self.setTitle('数据验证')
        self.setBorderRadius(8)

class ValView(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QWidget(self)

        self.vBoxLayout = QVBoxLayout(self.view)
        self.valMainInfoCard = ValMainInfoCard(self)
        self.descriptionCard = DescriptionCard(self)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setObjectName("valView")

        self.vBoxLayout.setSpacing(10)
        self.vBoxLayout.setContentsMargins(0, 0, 10, 30)
        self.vBoxLayout.addWidget(self.valMainInfoCard, 0, Qt.AlignTop)
        self.vBoxLayout.addWidget(self.descriptionCard, 0, Qt.AlignTop)
        self.enableTransparentBackground()
