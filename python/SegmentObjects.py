from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('weights/yolo11n-seg-coco.pt')
    model.eval()

    results = model('/home/alex/Projects/MediumBlog/YoloOnnxRuntimeCPP/test_images/donut3.jpg')

    for result in results:
        result.show()
        result.save('YoloObjectSegmentationResults1.jpg')
