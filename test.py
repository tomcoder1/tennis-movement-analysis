from ultralytics import YOLO

model = YOLO('siu.pt')

result = model.predict('full-match-data/short-clip.mp4', save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)