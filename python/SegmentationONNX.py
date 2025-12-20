import onnxruntime as ort
import numpy as np
import cv2
import time
from pathlib import Path

class BoundingBox:
    def __init__(self, x, y, w, h, class_id, confidence_score, mask_coefficients):
        self.xywh = [x, y, w, h]
        self.class_id = class_id
        self.confidence_score = confidence_score
        self.mask_coefficients = mask_coefficients
        self.mask = None

def preprocess_img(img: np.ndarray) -> np.ndarray:
    # img = cv2.resize(img, (640, 640))
    img = resize_letter_box(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, 640, 640)
    img = img / 255.0
    img = img.astype(np.float32)
    return img

def resize_letter_box(img: np.ndarray, new_size: tuple) -> np.ndarray:
    img_width, img_height = img.shape[1], img.shape[0]
    new_width, new_height = new_size[0], new_size[1]
    scale_ratio = min(new_width / img_width, new_height / img_height)

    new_unpad_width = int(round(img_width * scale_ratio))
    new_unpad_height = int(round(img_height * scale_ratio))

    dw = (new_width - new_unpad_width) / 2
    dh = (new_height - new_unpad_height) / 2

    if new_width != new_unpad_width or new_height != new_unpad_height:
        img = cv2.resize(img, (new_unpad_width, new_unpad_height), interpolation=cv2.INTER_LANCZOS4)

    pad_top = round(dh - 0.1)
    pad_bottom = round(dh + 0.1)
    pad_left = round(dw - 0.1)
    pad_right = round(dw + 0.1)
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, None,(114, 114, 114))

def get_filtered_boxes(boxes: np.ndarray):
    formatted_boxes = []
    for box in boxes:
        scores = box[4:84]
        class_id = int(scores.argmax())
        confidence_score = float(scores[class_id])
        if confidence_score > 0.25:
            w = int(box[2])
            h = int(box[3])
            # get x and y moving from center to upper left of box
            x = int(max(box[0] - 0.5 * w, 0))
            y = int(max(box[1] - 0.5 * h, 0))
            mask_coefficients = box[84:]
            new_box = BoundingBox(x, y, w, h, class_id, confidence_score, mask_coefficients)
            formatted_boxes.append(new_box)
    return formatted_boxes

def get_filtered_boxes_fast(boxes: np.ndarray) -> list[BoundingBox]:
    coords = boxes[:, :4, 0]
    class_scores = boxes[:, 4:84, 0]
    mask_coefficients = boxes[:, 84:, 0]

    # computer class + confidence
    class_ids = class_scores.argmax(axis=1)
    confidence_scores = class_scores[np.arange(len(class_ids)), class_ids]
    keep = confidence_scores > 0.25

    if not np.any(keep):
        return []

    coords = coords[keep]
    class_ids = class_ids[keep]
    confidence_scores = confidence_scores[keep]
    mask_coefficients = mask_coefficients[keep]

    # move x, y to top left
    widths = coords[:, 2].astype(np.int32)
    heights = coords[:, 3].astype(np.int32)

    xs = np.maximum(0, coords[:, 0] - 0.5 * widths).astype(np.int32)
    ys = np.maximum(0, coords[:, 1] - 0.5 * heights).astype(np.int32)

    return [
        BoundingBox(int(x), int(y), int(w), int(h), class_id, confidence_score, mask_coeff)
        for x, y, w, h, class_id, confidence_score, mask_coeff in zip(
            xs, ys, widths, heights, class_ids, confidence_scores, mask_coefficients
        )
    ]

def clip_box(box: BoundingBox, orig_img_size: tuple) -> BoundingBox:
    orig_img_width, orig_img_height = orig_img_size
    x, y, w, h = box.xywh
    x = max(0, min(x, orig_img_width))
    y = max(0, min(y, orig_img_height))
    w = max(0, min(w, orig_img_width - x))
    h = max(0, min(h, orig_img_height - y))
    box.xywh = [x, y, w, h]

def scale_boxes(boxes: list[BoundingBox], orig_image_size: tuple, letterbox_shape) -> list[BoundingBox]:
    orig_img_width, orig_img_height = orig_image_size
    letterbox_width, letterbox_height = letterbox_shape
    scale_ratio = min(letterbox_height / orig_img_height, letterbox_width / orig_img_width)
    horizontal_padding = (letterbox_width - orig_img_width * scale_ratio) / 2
    vertical_padding = (letterbox_height - orig_img_height * scale_ratio) / 2
    for box in boxes:
        x, y, w, h = box.xywh
        x -= horizontal_padding
        y -= vertical_padding
        x /= scale_ratio
        y /= scale_ratio
        w /= scale_ratio
        h /= scale_ratio
        box.xywh = [x, y, w, h]
    return boxes

def nms_boxes(filtered_boxes: list[BoundingBox]) -> list[BoundingBox]:
    boxes = []
    scores = []
    for box in filtered_boxes:
        boxes.append(box.xywh)
        scores.append(box.confidence_score)
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.6)
    filtered_boxes = [filtered_boxes[idx] for idx in indices]
    return filtered_boxes

def resize_mask_remove_letterbox(mask, original_img_size, letterbox_size):
    letterbox_width, letterbox_height = letterbox_size
    orig_img_width, orig_img_height = original_img_size
    scale_ratio = min(
        letterbox_height / orig_img_height, # height
        letterbox_width / orig_img_width) # width

    unpad_width = int(round(orig_img_width * scale_ratio))
    unpad_height = int(round(orig_img_height * scale_ratio))

    pad_width = letterbox_width - unpad_width
    pad_height = letterbox_height - unpad_height
    pad_left = int(round(pad_width / 2))
    pad_top = int(round(pad_height / 2))

    resized_mask = cv2.resize(mask, letterbox_size, interpolation=cv2.INTER_LINEAR)
    cropped_mask = resized_mask[pad_top:pad_top+unpad_height, pad_left:pad_left+unpad_width]
    return cv2.resize(cropped_mask, original_img_size, interpolation=cv2.INTER_LINEAR)

def process_bounding_boxes(boxes: np.ndarray, img_size) -> list[BoundingBox]:
    start = time.perf_counter()
    # filtered_bounding_boxes = get_filtered_boxes(boxes)
    filtered_bounding_boxes = get_filtered_boxes_fast(boxes)
    print(f'\tFiltering time: {(time.perf_counter()-start)*1000:.0f} ms')
    start = time.perf_counter()
    filtered_bounding_boxes = nms_boxes(filtered_bounding_boxes)
    print(f'\tNMS time: {(time.perf_counter()-start)*1000:.0f} ms')
    start = time.perf_counter()
    scaled_boxes = scale_boxes(filtered_bounding_boxes, img_size, (640, 640))
    print(f'\tScaling time: {(time.perf_counter()-start)*1000:.0f} ms')
    return scaled_boxes

def process_masks(proto_masks: np.ndarray, bboxes: list[BoundingBox], img_size: tuple) -> list[BoundingBox]:
    mask_width, mask_height = proto_masks.shape[2:]
    proto_masks = proto_masks.reshape((32, mask_width * mask_height))
    width, height = img_size
    # Iterates boxes; extracts and applies mask; thresholds mask
    for box in bboxes:
        mask_coefficients = box.mask_coefficients
        combined_mask = mask_coefficients.transpose() @ proto_masks
        combined_mask = combined_mask.reshape((mask_height, mask_width))
        resized_cropped_mask = resize_mask_remove_letterbox(
            combined_mask,
            (width, height),
            (640, 640))
        x, y, w, h = box.xywh
        cropped_mask = resized_cropped_mask[int(y):int(y+h), int(x):int(x+w)]
        _, cropped_mask = cv2.threshold(cropped_mask, 0.5, 255, cv2.THRESH_BINARY)
        box.mask = cropped_mask
    return bboxes

def draw_boxes(img: np.ndarray, boxes: list[BoundingBox]) -> np.ndarray:
    # Draws boxes and overlays masks onto the image
    for box in boxes:
        x, y, w, h = box.xywh
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        # Draw Mask
        roi = img[int(y):int(y+h), int(x):int(x+w)]
        mask_bool = (box.mask == 255)
        color_img = np.zeros_like(roi, dtype=np.uint8)
        color_img[:] = (0, 0, 255)
        alpha = 0.5
        blended_roi = cv2.addWeighted(roi, 1 - alpha, color_img, alpha, 0)
        roi[mask_bool] = blended_roi[mask_bool]
        img[int(y):int(y+h), int(x):int(x+w)] = roi
    return img

if __name__ == "__main__":
    model_path = 'weights/yolo11n-seg-coco.onnx'
    video_path = 'videos/baseball.mp4'
    output_path = f'videos/processed/python/{video_path.split("/")[-1]}'
    Path('videos/processed/python').mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    orig_img_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_img_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_img_width, orig_img_height))
    ort_session =  ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    io_binding = ort_session.io_binding()

    # performance stats
    avg_loop_time = 0
    avg_preprocess_time_ms = 0
    avg_inference_time_ms = 0
    avg_box_processing_time_ms = 0
    avg_mask_processing_time_ms = 0
    avg_drawing_time_ms = 0
    frames_processed = 0

    while video.isOpened():
        start_time = time.perf_counter()
        ret, frame = video.read()
        if not ret:
            break

        # Preprocessing
        s = time.perf_counter()
        input_img = preprocess_img(frame)
        ort_value = ort.OrtValue.ortvalue_from_numpy(input_img, 'cuda', device_id=0)
        # io_binding.bind_cpu_input('images', input_img)
        io_binding.bind_input(
            name='images',
            device_type=ort_value.device_name(),
            device_id=0,
            element_type=np.float32,
            shape=input_img.shape,
            buffer_ptr=ort_value.data_ptr()
        )
        io_binding.bind_output('output0')
        io_binding.bind_output('output1')
        time_diff = time.perf_counter() - s
        print(f'Preprocessing time: {time_diff * 1000:.0f} ms')
        avg_preprocess_time_ms += time_diff * 1000

        # Inference
        s = time.perf_counter()
        ort_session.run_with_iobinding(io_binding)
        time_diff = time.perf_counter() - s
        print(f'Inference time: {time_diff*1000:.0f} ms')
        avg_inference_time_ms += time_diff * 1000

        # Bounding box processing
        s = time.perf_counter()
        outputs = io_binding.copy_outputs_to_cpu()
        bounding_boxes = outputs[0].transpose()
        bounding_boxes = process_bounding_boxes(bounding_boxes, (orig_img_width, orig_img_height))
        time_diff = time.perf_counter() - s
        print(f'Box Processing: {time_diff*1000:.0f} ms')
        avg_box_processing_time_ms += time_diff * 1000

        # Mask processing
        s = time.perf_counter()
        masks = outputs[1]
        masks = process_masks(masks, bounding_boxes, (orig_img_width, orig_img_height))
        time_diff = time.perf_counter() - s
        print(f'Mask Processing: {time_diff*1000:.0f} ms')
        avg_mask_processing_time_ms += time_diff * 1000

        # Drawing boxes
        s = time.perf_counter()
        image = draw_boxes(frame, bounding_boxes)
        time_diff = time.perf_counter() - s
        print(f'Drawing boxes: {time_diff*1000:.0f} ms')
        avg_drawing_time_ms += time_diff * 1000

        frames_processed += 1
        end_time = time.perf_counter()
        avg_loop_time += end_time - start_time
        cv2.imshow('image', image)
        cv2.waitKey(1)
        writer.write(image)
        print('-' * 10)

    video.release()
    cv2.destroyAllWindows()
    avg_loop_time = avg_loop_time / frames_processed
    print(f"Average processing time: {avg_preprocess_time_ms / frames_processed:.0f} ms")
    print(f"Average inference time: {avg_inference_time_ms / frames_processed:.0f} ms")
    print(f"Average box processing time: {avg_box_processing_time_ms / frames_processed:.0f} ms")
    print(f"Average mask processing time: {avg_mask_processing_time_ms / frames_processed:.0f} ms")
    print(f"Average drawing time: {avg_drawing_time_ms / frames_processed:.0f} ms")
    print(f"Average loop time: {avg_loop_time * 1000:.0f} ms")
    print(f"Average fps: {1 / avg_loop_time:.0f} fps")