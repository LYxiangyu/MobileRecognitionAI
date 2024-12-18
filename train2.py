import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
import numpy as np
from sklearn.metrics import accuracy_score
import logging

# 定义一个简化的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 减少卷积层的通道数
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 将 32 减少到 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 将 64 减少到 32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 调整全连接层的输入尺寸
        self.fc1 = nn.Linear(32 * 160 * 160, 256)  # 将 512 减少到 256
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 160 * 160)  # 调整为 32 * 160 * 160
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(data_dir, target_class)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                path = os.path.join(class_dir, file_name)
                if os.path.isfile(path):
                    self.samples.append((path, self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(val_loader), correct / total


# 测试函数
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # 保持原始图像尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载训练集、验证集和测试集
    train_dataset = CustomDataset('./dataset/train', transform=transform)
    val_dataset = CustomDataset('./dataset/val', transform=transform)
    test_dataset = CustomDataset('./dataset/test', transform=transform)

    # 减少批量大小
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)  # 将 batch_size 从 4 减少到 2
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=8)

    # 初始化模型、损失函数和优化器
    num_classes = len(train_dataset.classes)
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10
    best_val_acc = 0.0
    best_model_path = './runs/classify/train2/weights/best2.pth'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(num_epochs):
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}]')

        # 训练
        train_loss = train(model, train_loader, criterion, optimizer, device)
        logging.info(f'Train Loss: {train_loss:.4f}')

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'Saved best model with validation accuracy: {best_val_acc:.4f}')

    # 测试模型
    model.load_state_dict(torch.load(best_model_path))
    test_acc = test(model, test_loader, device)
    logging.info(f'Test Acc: {test_acc:.4f}')

    # 保存类别名称到文件
    with open('./runs/classify/train2/classes.txt', 'w') as f:
        for cls_name in train_dataset.classes:
            f.write(f"{cls_name}\n")
    logging.info("Saved class names to classes.txt")