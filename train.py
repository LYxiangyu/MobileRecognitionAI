import torch
from ultralytics import YOLO

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 打印选择的设备
    print(f"Using device: {device}")
    model = YOLO('yolo11n-cls.pt')
    model.train(data='./dataset',epochs = 100,batch = 4, workers = 8,imgsz =640,device=device)
