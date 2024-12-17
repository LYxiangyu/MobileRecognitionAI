from ultralytics import YOLO
import numpy as np

if __name__ == '__main__':
    model = YOLO("./runs/classify/train/weights/best.pt")
    results = model('dataset/1734425537672.jpg',show = True)
    names_dict = results[0].names

    probs = results[0].probs.data.tolist()

    print(names_dict)
    print(probs)
    print(names_dict[np.argmax(probs)])