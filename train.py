from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')
    model.train(data='./dataset/data.yml',epochs = 100,batch = 16, workers = 8,imgsz =640)
