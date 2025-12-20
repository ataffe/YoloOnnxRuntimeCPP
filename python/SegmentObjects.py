from ultralytics import YOLO
import cv2
import numpy as np
import time

if __name__ == '__main__':
    model = YOLO('weights/yolo11n-seg-coco.pt').cuda()
    model.eval()

    cap = cv2.VideoCapture('videos/baseball.mp4')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('videos/baseball_python.mp4', fourcc, 30.0, (width, height))
    average_loop_time = 0
    iterations = 0
    while cap.isOpened():
        start_time = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()
        results = model(frame)
        segmented_img = frame.copy()
        for result in results:
            if result is not None and result.masks is not None:

                alpha = 0.5
                segmented_img = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        end_time = time.perf_counter()
        average_loop_time += end_time - start_time
        iterations += 1
        cv2.imshow('frame', segmented_img)
        out.write(segmented_img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    out.release()
    average_loop_time /= iterations
    print(f"Average loop time: {average_loop_time * 1000:.2f} ms")
    print(f"Average fps: {1 / average_loop_time:.2f} fps")

