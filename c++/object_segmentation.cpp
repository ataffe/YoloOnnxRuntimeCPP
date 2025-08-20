//
// Created by alex on 8/19/25.
//
//
// Created by alex on 8/8/25.
//
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "letter_box.h"
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <iostream>
#include <chrono>

struct YoloBoundingBox {
    cv::Rect bounding_box;
    double confidence;
    int class_id;
    std::vector<float> mask_coefficients; // <-- NEW
    cv::Mat mask; // <-- NEW
};

std::map<int, std::string> ReadCOCOLabels(const std::string &labels_path) {
    std::ifstream labels_file(labels_path);
    if (!labels_file.is_open()) {
        std::cerr << "Error opening file\n";
        throw std::runtime_error("Error opening file");
    }
    std::string word;
    std::map<int, std::string> labels;
    int line_number = 0;
    while (std::getline(labels_file, word)) {
        labels[line_number] = word;
        line_number++;
    }
    return labels;
}


Ort::Session LoadYoloModel(const Ort::Env &env, const std::string &model_path,
                           OrtCUDAProviderOptionsV2 *cuda_provider) {
    Ort::SessionOptions yolo_session_options;
    Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_provider));
    yolo_session_options.AppendExecutionProvider_CUDA_V2(*cuda_provider);
    Ort::Session yolo_model_session(env, model_path.c_str(), yolo_session_options);
    return yolo_model_session;
}

cv::Mat ImageToBlob(const cv::Mat &image) {
    cv::Mat resized_image;
    LetterBox(image, resized_image, cv::Size(640, 640));
    cv::imwrite("LetterBoxed.jpg", resized_image);
    return cv::dnn::blobFromImage(
        resized_image,
        1.0 / 255.0,
        cv::Size(640, 640),
        cv::Scalar::all(0),
        true,
        false,
        CV_32F);
}

Ort::Value BlobToONNXTensor(const cv::Mat &blob) {
    std::vector<int64_t> tensor_shape = {1, 3, 640, 640};
    int64_t blob_size = 3 * 640 * 640;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(
        memory_info,
        (float *) (blob.data), // C++ style casting will break this.
        blob_size,
        tensor_shape.data(),
        tensor_shape.size()
    );
}

cv::Mat GetYoloBoxes(std::vector<Ort::Value> &output) {
    std::vector<int64_t> output_shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
    // [batch size, mask weights, boxes predicted]=>[bs, preds_num, features]
    return cv::Mat(
        cv::Size(static_cast<int>(output_shape[2]),
                 static_cast<int>(output_shape[1])),
        CV_32F,
        output[0].GetTensorMutableData<float>()).t();
}

void ClipBox(cv::Rect &box, const cv::Size &shape) {
    box.x = std::max(0, std::min(box.x, shape.width));
    box.y = std::max(0, std::min(box.y, shape.height));
    box.width = std::max(0, std::min(box.width, shape.width - box.x));
    box.height = std::max(0, std::min(box.height, shape.height - box.y));
}

void ScaleYoloBoundingBox(YoloBoundingBox &box, const cv::Size &original_shape) {
    cv::Size yolo_shape(640, 640);
    float scale_ratio = std::min(static_cast<float>(yolo_shape.width) / static_cast<float>(original_shape.width),
                                 static_cast<float>(yolo_shape.height) / static_cast<float>(original_shape.height));
    float letterbox_pad_horizontal = (yolo_shape.width - original_shape.width * scale_ratio) / 2;
    float letterbox_pad_vertical = (yolo_shape.height - original_shape.height * scale_ratio) / 2;
    box.bounding_box.x -= letterbox_pad_horizontal;
    box.bounding_box.y -= letterbox_pad_vertical;
    box.bounding_box.x /= scale_ratio;
    box.bounding_box.y /= scale_ratio;
    box.bounding_box.width /= scale_ratio;
    box.bounding_box.height /= scale_ratio;
    ClipBox(box.bounding_box, original_shape);
}

cv::Mat ResizeMaskRemoveLetterbox(const cv::Mat &input_mask, const cv::Size &orig_image_shape,
                                  const cv::Size &letterbox_shape) {
    // Calculate scale ratio
    float scale_ratio = std::min(
        static_cast<float>(letterbox_shape.height) / static_cast<float>(orig_image_shape.height),
        static_cast<float>(letterbox_shape.width) / static_cast<float>(orig_image_shape.width));

    // Calculate unpadded letterboxed image width / height
    int unpadded_width = static_cast<int>(std::round(orig_image_shape.width * scale_ratio));
    int unpadded_height = static_cast<int>(std::round(orig_image_shape.height * scale_ratio));

    // Calculate horizontal / vertical padding
    int pad_w = letterbox_shape.width - unpadded_width;
    int pad_h = letterbox_shape.height - unpadded_height;

    // Calculate left and top
    int pad_left = static_cast<int>(std::round(pad_w / 2.0f));
    int pad_top = static_cast<int>(std::round(pad_h / 2.0f));

    // Step 2: Resize mask to letterbox shape (from 160x160 to 640x640)
    cv::Mat resized_mask;
    cv::resize(input_mask, resized_mask, letterbox_shape, 0, 0, cv::INTER_LINEAR);

    // Step 3: Crop out the padding area
    cv::Rect roi(pad_left, pad_top, unpadded_width, unpadded_height);
    cv::Mat cropped_mask = resized_mask(roi);

    // Step 4: Resize to desired final output size (640x640)
    cv::Mat final_mask;
    cv::resize(cropped_mask, final_mask, orig_image_shape, 0, 0, cv::INTER_LINEAR);

    return final_mask;
}

std::vector<YoloBoundingBox> ProcessYoloBoundingBoxes(const cv::Mat &raw_boxes, const cv::Size &original_shape) {
    int num_classes = 80;
    int num_mask_coeffs = 32; // <-- NEW
    int data_width = num_classes + num_mask_coeffs + 4; // 4 is for x, y, width, height
    auto *bounding_box_data = reinterpret_cast<float *>(raw_boxes.data);
    std::vector<YoloBoundingBox> processed_boxes;
    for (int row = 0; row < raw_boxes.rows; row++) {
        // Step 1: Parse x, y, Width, Height, max class score
        // Save all class scores 1 - 80 starting at index 4
        cv::Mat class_scores = cv::Mat(1, num_classes, CV_32FC1, bounding_box_data + 4);
        cv::Point class_id_tmp;
        double confidence;
        // Find max score
        cv::minMaxLoc(class_scores, nullptr, &confidence, nullptr, &class_id_tmp);
        // Step 2: Filter boxes by confidence
        if (confidence > 0.25) {
            float width = bounding_box_data[2];
            float height = bounding_box_data[3];
            float x = MAX((bounding_box_data[0] - 0.5 * width), 0);
            float y = MAX((bounding_box_data[1] - 0.5 * height), 0);
            cv::Rect bounding_box(x, y, width, height);
            int class_id = class_id_tmp.x;
            std::vector mask_coefficients(bounding_box_data + 4 + num_classes, bounding_box_data + data_width);
            // <- NEW
            YoloBoundingBox parsed_box(bounding_box, confidence, class_id, mask_coefficients);
            processed_boxes.push_back(parsed_box);
        }
        // Advance data pointer to next row
        bounding_box_data += data_width;
    }
    // Step 3: Scale boxes
    for (auto &box: processed_boxes) {
        ScaleYoloBoundingBox(box, original_shape);
    }
    // Step 4: Non-Max Suppression
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> filtered_idxs;
    for (auto &processed_box: processed_boxes) {
        boxes.push_back(processed_box.bounding_box);
        confidences.push_back(processed_box.confidence);
    }
    cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.70, filtered_idxs);
    std::vector<YoloBoundingBox> filtered_boxes;
    for (auto &idx: filtered_idxs) {
        filtered_boxes.push_back(processed_boxes[idx]);
    }
    return filtered_boxes;
}

void ProcessYOLOMasks(std::vector<YoloBoundingBox> &boxes, const cv::Mat &raw_proto_masks, cv::Size orig_img_size) {
    int proto_mask_width = raw_proto_masks.size[2];
    int proto_mask_height = raw_proto_masks.size[3];

    // Reshape from 4d: [1, 32, 160, 160] to 2d: [32, 160 * 160]
    cv::Mat proto_masks = raw_proto_masks.clone().reshape(0, {
                                                              32,
                                                              proto_mask_width * proto_mask_height
                                                          });
    for (auto &box: boxes) {
        // Convert coefficients to matrix and Combine mask prototypes and coefficients
        cv::Mat combined_masks = cv::Mat(box.mask_coefficients).t() * proto_masks;
        // Reshape to 2d: [1, 160 * 160]
        combined_masks = combined_masks.reshape(1, {proto_mask_width, proto_mask_height});
        // Scale back to original size and remove letterbox padding
        cv::Mat scaled_mask = ResizeMaskRemoveLetterbox(combined_masks, orig_img_size, cv::Size(640, 640));
        // Crop to bounding box
        cv::Mat cropped_mask = cv::Mat::zeros(scaled_mask.size(), scaled_mask.type());
        scaled_mask(box.bounding_box).copyTo(cropped_mask(box.bounding_box));
        // Convert to binary mask
        cropped_mask.convertTo(box.mask, CV_8UC1, 255);
    }
}

void DrawBoxes(const std::vector<YoloBoundingBox> &boxes, const cv::Mat &image) {
    std::map<int, std::string> labels = ReadCOCOLabels(
        "labels/coco.txt");
    for (const auto &box: boxes) {
        // Draw bounding box
        cv::rectangle(image, box.bounding_box, cv::Scalar(0, 255, 0), 2);
        // Draw label
        std::string label = labels[box.class_id];
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << box.confidence;
        label += ": " + oss.str();
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(image, cv::Point(box.bounding_box.x, box.bounding_box.y - label_size.height),
                      cv::Point(box.bounding_box.x + label_size.width, box.bounding_box.y),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(image, label, cv::Point(box.bounding_box.x, box.bounding_box.y - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        // Draw mask
        cv::Mat mask = image.clone();
        auto mask_color = cv::Scalar(0, 0, 255);
        mask(box.bounding_box).setTo(mask_color, box.mask(box.bounding_box));
        cv::addWeighted(image, 0.5, mask, 0.5, 0, image);
    }
}

cv::Mat getYoloProtoMasks(std::vector<Ort::Value> &detection) {
    auto proto_mask_shape = detection[1].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int> mask_sz = {
        1,
        static_cast<int>(proto_mask_shape[1]), // mask weights
        static_cast<int>(proto_mask_shape[2]), // mask height
        static_cast<int>(proto_mask_shape[3]) // mask width
    };
    return {mask_sz, CV_32F, detection[1].GetTensorMutableData<float>()};
}

int main() {
    // Create environment
    const Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "Yolo Tutorial");
    OrtCUDAProviderOptionsV2 *cuda_provider = nullptr;
    // Load Model
    Ort::Session yolo_model_session = LoadYoloModel(
        env, "weights/yolo11n-seg-coco.onnx", cuda_provider);
    // Read image
    cv::Mat image = cv::imread("test_images/ex1_coco2017.jpg");
    // Convert image to blob
    cv::Mat blob = ImageToBlob(image);
    // Convert blob to tensor
    Ort::Value input_tensor = BlobToONNXTensor(blob);
    // Get Input Names
    Ort::AllocatorWithDefaultOptions allocator;
    const std::string input_name = yolo_model_session.GetInputNameAllocated(0, allocator).get();
    const std::string output_name1 = yolo_model_session.GetOutputNameAllocated(0, allocator).get();
    const std::string output_name2 = yolo_model_session.GetOutputNameAllocated(1, allocator).get();
    const char *input_names[] = {input_name.c_str()};
    const char *output_names[] = {output_name1.c_str(), output_name2.c_str()};

    // Run inference
    std::vector<Ort::Value> output = yolo_model_session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        2
    );
    // Get bounding boxes as a Mat object
    cv::Mat raw_boxes = GetYoloBoxes(output);
    // Get masks as a Mat object
    cv::Mat prototype_masks = getYoloProtoMasks(output);
    // Process bounding boxes
    std::vector<YoloBoundingBox> boxes = ProcessYoloBoundingBoxes(raw_boxes, image.size());
    // Process segmentation masks
    ProcessYOLOMasks(boxes, prototype_masks, image.size());
    // Draw boxes on image
    DrawBoxes(boxes, image);
    cv::imshow("Test Image", image);
    cv::imwrite("output.jpg", image);
    cv::waitKey(0);
    Ort::GetApi().ReleaseCUDAProviderOptions(cuda_provider);
    return 0;
}
