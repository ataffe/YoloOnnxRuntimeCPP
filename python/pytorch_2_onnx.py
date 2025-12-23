from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    print('Converting YOLO model to ONNX...')
    model = YOLO(args.model_path)
    model.eval()
    model.export(format='onnx', simplify=True)