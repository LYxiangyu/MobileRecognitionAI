from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolo11n-cls.pt')
    model.train(data='./dataset',epochs = 100,batch = 4, workers = 8,imgsz =640,device='0')
