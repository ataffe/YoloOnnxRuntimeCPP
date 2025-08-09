from ultralytics import YOLO

model = YOLO('../weights/yolo11s_imagenet.pt')
model.eval()

results = model('../test_images/ExampleImageUnsplash.jpg')

for result in results:
    result.show()
    result.save('YoloObjectDetectionResults2.jpg')