import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
import argparse
import os
from train2 import SimpleCNN  # 直接从 train2.py 导入 SimpleCNN 模型
def load_classes(file_path):
    """加载类别名称"""
    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def predict(image_path, model, transform, device, classes):
    """对单张图片进行分类预测"""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)
        prob = probs[0, pred].item()
    return classes[pred.item()], prob


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Image classification prediction')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 加载类别名称
    classes_file = './runs/classify/train2/classes.txt'
    if not os.path.exists(classes_file):
        logging.error(f"Classes file not found at {classes_file}")
        exit(1)
    classes = load_classes(classes_file)
    num_classes = len(classes)

    # 加载模型
    model = SimpleCNN(num_classes=num_classes).to(device)
    model_path = './runs/classify/train2/weights/best2.pth'
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        exit(1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logging.info(f"Loaded model from {model_path}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to 640x640
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 进行预测
    prediction, probability = predict(args.image_path, model, transform, device, classes)
    logging.info(f"Predicted class: {prediction}, Probability: {probability:.4f}")

    print(f"Predicted class: {prediction}, Probability: {probability:.4f}")