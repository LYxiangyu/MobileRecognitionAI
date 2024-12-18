import sys

import torch
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import QApplication
from ultralytics import YOLO

def startTraining():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 打印选择的设备
    print(f"Using device: {device}")
    model = YOLO('yolo11n-cls.pt')
    model.train(data='./dataset', epochs=10, batch=4, workers=8, imgsz=640, device=device)

# 创建一个子线程用于训练
class TrainingThread(QThread):
    training_started = pyqtSignal()  # 当训练开始时，发出信号
    training_finished = pyqtSignal()  # 当训练完成时，发出信号

    def run(self):
        try:
            # 训练任务
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
            model = YOLO('yolo11n-cls.pt')  # 加载模型
            model.train(data='./dataset', epochs=10, batch=4, workers=8, imgsz=640, device=device)
            self.training_finished.emit()  # 训练完成后发出信号
        except Exception as e:
            print(f"Error during training: {e}")
            self.training_finished.emit()  # 即使出错，也发出训练完成信号