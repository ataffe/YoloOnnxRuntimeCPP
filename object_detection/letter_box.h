//
// Created by alex on 8/13/25.
//

#ifndef YOLOONNXC_TUTORIAL_LETTERBOX_H
#define YOLOONNXC_TUTORIAL_LETTERBOX_H
#include <cmath>
#include <opencv2/opencv.hpp>

const int &DEFAULT_LETTERBOX_PAD_VALUE = 114;

void LetterBox(const cv::Mat &image, cv::Mat &output_image, const cv::Size &new_shape) {
    float scale_ratio = std::min(static_cast<float>(new_shape.height) / static_cast<float>(image.size().height),
                                 static_cast<float>(new_shape.width) / static_cast<float>(image.size().width));

    int new_unpad[2] = {
        static_cast<int>(std::round(image.size().width * scale_ratio)),
        static_cast<int>(std::round(image.size().height * scale_ratio))
    };

    int dw = (new_shape.width - new_unpad[0]) / 2;
    int dh = (new_shape.height - new_unpad[1]) / 2;

    if (new_shape.width != new_unpad[0] || new_shape.height != new_unpad[1]) {
        cv::resize(image, output_image, cv::Size(new_unpad[0], new_unpad[1]), 0, 0, cv::INTER_LANCZOS4);
    } else {
        output_image = image.clone();
    }

    int pad_top = static_cast<int>(std::round(dh - 0.1f));
    int pad_bottom = static_cast<int>(std::round(dh + 0.1f));
    int pad_left = static_cast<int>(std::round(dw - 0.1f));
    int pad_right = static_cast<int>(std::round(dw + 0.1f));
    cv::copyMakeBorder(output_image, output_image, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT,
                       cv::Scalar(DEFAULT_LETTERBOX_PAD_VALUE, DEFAULT_LETTERBOX_PAD_VALUE,
                                  DEFAULT_LETTERBOX_PAD_VALUE));
}
#endif //YOLOONNXC_TUTORIAL_LETTERBOX_H