import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import copy
import torch.nn.functional as F
import logging
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 数据加载和预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def prepare_data(data_dir="./dataset"):
    image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print("Class names:", class_names)
    return dataloaders, dataset_sizes, class_names

# 2. 设置迁移学习模型和优化器
def setup_model_and_optimizer(num_classes):
    # 加载预训练的 ResNet-152 模型
    model = models.resnet152(pretrained=True)  # 修改为 ResNet-152

    # 冻结所有层的参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后一层全连接层以适应新的类别数
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 只对新添加的层进行训练
    for param in model.fc.parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-5)  # 只优化新添加的层
    return model, criterion, optimizer, device


# 3. 早停法类
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# 4. 训练模型
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=30, early_stopping=None):
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    scaler = GradScaler()

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} Epoch [{epoch + 1}]"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():  # 使用混合精度训练
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                history['loss'].append(epoch_loss)
                history['accuracy'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_accuracy'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val' and early_stopping is not None:
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # 更新学习率
            if phase == 'val':
                scheduler.step(epoch_loss)

        if early_stopping is not None and early_stopping.early_stop:
            break

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history


# 5. 主函数
def main():
    # 确保目录存在
    os.makedirs('./runs/classify/train2/weights', exist_ok=True)

    # 准备数据
    dataloaders, dataset_sizes, class_names = prepare_data(data_dir="./dataset")

    # 设置迁移学习模型和优化器
    model, criterion, optimizer, device = setup_model_and_optimizer(num_classes=len(class_names))

    # 早停法实例
    early_stopping = EarlyStopping(patience=10, verbose=True, path='./runs/classify/train2/weights/best3.pth')

    # 训练模型
    model, history = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=30,
                                 early_stopping=early_stopping)

    # 保存模型
    torch.save(model.state_dict(), './runs/classify/train2/weights/best3.pth')

    # 保存类别名称到文件
    with open('./runs/classify/train2/classes.txt', 'w') as f:
        for cls_name in class_names:
            f.write(f"{cls_name}\n")
    logging.info("Saved class names to classes.txt")


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()