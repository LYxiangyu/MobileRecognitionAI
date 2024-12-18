import torch
from torchvision import transforms
from PIL import Image
import os
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from train2 import SimpleCNN  # 直接从 train2.py 导入 SimpleCNN 模型

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    return pred.item()


def evaluate_model(test_dir, model, transform, device, classes):
    """评估模型在测试集上的性能"""
    true_labels = []
    predicted_labels = []

    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_idx = classes.index(class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            if os.path.isfile(image_path):
                pred_label = predict(image_path, model, transform, device, classes)
                true_labels.append(class_idx)
                predicted_labels.append(pred_label)

    # 计算各类指标
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1


if __name__ == '__main__':
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
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logging.info(f"Loaded model from {model_path}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to 640x640
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 测试集路径
    test_dir = './dataset/test'  # 替换为你的测试集路径

    # 评估模型
    accuracy, precision, recall, f1 = evaluate_model(test_dir, model, transform, device, classes)

    # 打印评估结果
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")