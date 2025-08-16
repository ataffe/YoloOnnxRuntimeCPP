from ultralytics import YOLO

model = YOLO('/weights/yolo11n-detect-coco.pt')
model.eval()

results = model('/home/alex/Projects/MediumBlog/YoloOnnxRuntimeCPP/test_images/ex2_coco2017.jpg')

for result in results:
    result.show()
    result.save('YoloObjectDetectionResults2.jpg')