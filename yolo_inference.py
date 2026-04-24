from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.predict('input-data/image.png', save=True)