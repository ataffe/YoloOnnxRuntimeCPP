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

std::vector<YoloBoundingBox> ProcessYoloOutput(const cv::Mat &raw_boxes, const cv::Size &original_shape) {
    int num_classes = 80;
    int data_width = num_classes + 4;
    auto *bounding_box_data = reinterpret_cast<float *>(raw_boxes.data);
    std::vector<YoloBoundingBox> processed_boxes;
    for (int row = 0; row < raw_boxes.rows; row++) {
        // Step 1: Parse x, y, Width, Height, max class score
        float width = bounding_box_data[2];
        float height = bounding_box_data[3];
        float x = MAX((bounding_box_data[0] - 0.5 * width), 0);
        float y = MAX((bounding_box_data[1] - 0.5 * height), 0);
        cv::Rect bounding_box(x, y, width, height);
        // Save all class scores 1 - 80 starting at index 4
        cv::Mat class_scores = cv::Mat(1, num_classes, CV_32FC1, bounding_box_data + 4);
        cv::Point class_id_tmp;
        double confidence;
        // Find max score
        cv::minMaxLoc(class_scores, nullptr, &confidence, nullptr, &class_id_tmp);
        // Step 2: Filter boxes by confidence
        if (confidence > 0.25) {
            int class_id = class_id_tmp.x;
            YoloBoundingBox parsed_box(bounding_box, confidence, class_id);
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

int main() {
    const Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "Yolo Tutorial");
    OrtCUDAProviderOptionsV2 *cuda_provider = nullptr;
    Ort::Session yolo_model_session = LoadYoloModel(
        env, "weights/yolo11n-detect-coco.onnx", cuda_provider);
    cv::Mat image =
            cv::imread("test_images/ex2_coco2017.jpg");

    cv::Mat blob = ImageToBlob(image);
    Ort::Value input_tensor = BlobToONNXTensor(blob);
    Ort::AllocatorWithDefaultOptions allocator;
    const std::string input_name = yolo_model_session.GetInputNameAllocated(0, allocator).get();
    const std::string output_name = yolo_model_session.GetOutputNameAllocated(0, allocator).get();
    const char *input_names[] = {input_name.c_str()};
    const char *output_names[] = {output_name.c_str()};
    std::vector<Ort::Value> output = yolo_model_session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    cv::Mat raw_boxes = GetYoloBoxes(output);
    std::vector<YoloBoundingBox> boxes = ProcessYoloOutput(raw_boxes, image.size());


    std::map<int, std::string> labels = ReadCOCOLabels(
        "/labels/coco.txt");
    for (auto &box: boxes) {
        cv::rectangle(image, box.bounding_box, cv::Scalar(0, 255, 0), 2);
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
    }
    cv::imshow("Test Image", image);
    cv::imwrite("output.jpg", image);
    cv::waitKey(0);
    Ort::GetApi().ReleaseCUDAProviderOptions(cuda_provider);
    return 0;
}

