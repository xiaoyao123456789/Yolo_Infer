#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <opencv2/opencv.hpp>

void main_seg_run(const std::string& imagePath, const std::wstring& modelPath,
                  const std::string& outputPath);
void seg_nms(const std::vector<cv::Rect>& boxes,
             const std::vector<float>& scores, float score_threshold,
             float nms_threshold, std::vector<int>& indices);
void seg_scale_box(cv::Rect& box, cv::Size origin_size, float r, float dw,
                   float dh);
void post_process_seg(cv::Mat& origin, cv::Mat& result,
                      std::vector<Ort::Value>& outputs,
                      float r,   // r
                      float dw,  // dw
                      float dh);

static const cv::Scalar COLORS[] = {
    cv::Scalar(255, 56, 56),    // Red-ish
    cv::Scalar(255, 157, 151),  // Pink
    cv::Scalar(255, 112, 31),   // Orange
    cv::Scalar(255, 178, 29),   // Yellow
    cv::Scalar(207, 210, 49),   // Light yellow-green
    cv::Scalar(72, 250, 150),   // Green
    cv::Scalar(61, 220, 255),   // Light blue
    cv::Scalar(0, 133, 255),    // Blue
    cv::Scalar(0, 95, 255),     // Darker blue
    cv::Scalar(0, 0, 255)       // Pure red
};
static const int NUM_COLORS = sizeof(COLORS) / sizeof(COLORS[0]);
