import torch
from PyQt5.QtCore import pyqtSignal, Qt, QUrl, QThread
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import SimpleCardWidget, ImageLabel, TitleLabel, PrimaryPushButton, HyperlinkLabel, \
    VerticalSeparator, BodyLabel, PillPushButton, setFont, GroupHeaderCardWidget, PushButton, ComboBox, SearchLineEdit, \
    HeaderCardWidget, ScrollArea
from torchvision import transforms
from ultralytics import YOLO

from train2 import SimpleCNN
from val2 import evaluate_model, load_classes


class HandwrittenModelThread(QThread):
    finished_signal = pyqtSignal(float, float, float, float)  # 精确度、召回率等

    def __init__(self, test_dir, model, transform, device, classes):
        super().__init__()
        self.test_dir = test_dir
        self.model = model
        self.transform = transform
        self.device = device
        self.classes = classes

    def run(self):
        try:
            accuracy, precision, recall, f1 = evaluate_model(
                self.test_dir, self.model, self.transform, self.device, self.classes
            )
            self.finished_signal.emit(accuracy, precision, recall, f1)
        except Exception as e:
            print(f"Handwritten model evaluation failed: {str(e)}")


class YOLOModelThread(QThread):
    finished_signal = pyqtSignal(float, float)  # top1 和 top5 精确度

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            model = YOLO(self.model_path)
            metrics = model.val()
            self.finished_signal.emit(metrics.top1, metrics.top5)
        except Exception as e:
            print(f"YOLO model evaluation failed: {str(e)}")


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
        # 连接验证按钮
        self.valMainInfoCard.startButton.clicked.connect(self.start_validation)

        # 初始化两个线程
        self.handwritten_thread = None
        self.yolo_thread = None

        # 结果缓存
        self.yolo_result = ""
        self.handwritten_result = ""

    def start_validation(self):
        """开始两个模型的验证"""
        self.descriptionCard.descriptionLabel.setText("正在验证，请稍候...")

        # 加载手写模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classes = load_classes('./runs/classify/train2/classes.txt')
        model = SimpleCNN(num_classes=len(classes)).to(device)
        model.load_state_dict(torch.load('./runs/classify/train2/weights/best2.pth', map_location=device))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 手写模型验证线程
        self.handwritten_thread = HandwrittenModelThread(
            './dataset/test', model, transform, device, classes
        )
        self.handwritten_thread.finished_signal.connect(self.update_handwritten_result)
        self.handwritten_thread.start()

        # YOLO 模型验证线程
        self.yolo_thread = YOLOModelThread('./runs/classify/train/weights/best.pt')
        self.yolo_thread.finished_signal.connect(self.update_yolo_result)
        self.yolo_thread.start()

    def update_handwritten_result(self, accuracy, precision, recall, f1):
        """更新手写模型的验证结果"""
        self.handwritten_result = (
            f"手写模型验证结果：\n"
            f"准确率: {accuracy:.4f}\n"
            f"精确率: {precision:.4f}\n"
            f"召回率: {recall:.4f}\n"
            f"F1值: {f1:.4f}\n"
        )
        self.update_description_card()

    def update_yolo_result(self, top1, top5):
        """更新 YOLO 模型的验证结果"""
        self.yolo_result = (
            f"YOLO 模型验证结果：\n"
            f"Top1 精度: {top1:.4f}\n"
            f"Top5 精度: {top5:.4f}\n"
        )
        self.update_description_card()

    def update_description_card(self):
        """更新 DescriptionCard 显示的内容"""
        combined_result = f"{self.yolo_result}\n{self.handwritten_result}"
        self.descriptionCard.descriptionLabel.setText(combined_result.strip())
