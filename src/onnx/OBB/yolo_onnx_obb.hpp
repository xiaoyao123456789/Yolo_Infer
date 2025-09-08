/**************************************************
 * @file    yolo_onnx_obb.hpp
 * @brief   YOLOv8-OBB 头文件（声明 + 常量）
 * @author  姚
 * @date    2025-06-04
 **************************************************/
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

// ========================== 功能函数声明

float computeIoU(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);
void process(const wchar_t* model, std::vector<float>& inputs,
             std::vector<Ort::Value>& outputs);

void obb_nms(std::vector<cv::RotatedRect>& rboxes, std::vector<float>& scores,
             float score_threshold, float nms_threshold,
             std::vector<int>& indices);

void scale_box(cv::RotatedRect& rbox, cv::Size size);

void draw_result(cv::Mat& image, const std::string& label,
                 const cv::RotatedRect& rbox);

void post_process(cv::Mat& origin, cv::Mat& result,
                  std::vector<Ort::Value>& outputs);

void main_run(const std::string& imagePath, const std::wstring& modelPath,
              const std::string& outputPath);