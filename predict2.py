import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from train2 import SimpleCNN  # 导入训练时使用的模型

# 加载模型
model = SimpleCNN(num_classes=10)  # 确保类别数量与训练时一致
model.load_state_dict(torch.load('./runs/classify/train2/weights/best2.pth'))
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to 640x640
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 预测函数
def predict(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        prob, pred = torch.max(probs, 1)
    return pred.item(), prob.item()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = 'dataset/1734425537672.jpg'
    prediction, probability = predict(image_path, model, transform, device)
    print(f"Predicted class: {prediction}, Probability: {probability:.4f}")