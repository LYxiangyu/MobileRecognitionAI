from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("./runs/classify/train/weights/best.pt")  # load the trained model

    # Validate the model
    metrics = model.val()  # no arguments needed, uses the dataset and settings from training
    metrics.top1  # top1 accuracy
    metrics.top5  # top5 accuracy
    print("Validation Metrics:", metrics)