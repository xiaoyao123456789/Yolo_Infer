#pragma once
/**************************************************
 * @file    yolo_onnx_obb.hpp
 * @brief   YOLOv8-OBB 头文件（声明 + 常量）
 * @author  姚
 * @date    2025-06-04
 **************************************************/
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

// ================= 全局常量（C++17 起用 inline 避免多重定义） ================
inline const std::vector<std::string> class_names = {"Crack", "Patch"};
inline const int input_width = 1280;
inline const int input_height = 1280;
inline const int num_classes = static_cast<int>(class_names.size());
inline const int output_numprob = 5 + num_classes;
inline const int output_numbox = input_width / 8 * input_height / 8 +
                                 input_width / 16 * input_height / 16 +
                                 input_width / 32 * input_height / 32;
inline const float score_threshold = 0.25f;
inline const float nms_threshold = 0.5f;

// =========================== OrtSession 包装类 =============================
class OrtSessionWrapper {
 public:
  static Ort::Env& get_env();                               // 单例 Env
  explicit OrtSessionWrapper(const wchar_t* model_path);    // 构造
  void run_inference(const std::vector<float>& input_data,  // 推理接口
                     std::vector<Ort::Value>& outputs);

 private:
  void setup_io_info();  // 解析 I/O

  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_names_ptrs_;
  std::vector<const char*> output_names_ptrs_;
  std::vector<int64_t> input_dims_;
};

// ========================== 功能函数声明 ====================================
void process(const wchar_t* model, std::vector<float>& inputs,
             std::vector<Ort::Value>& outputs);

void nms(std::vector<cv::Rect>& boxes, std::vector<float>& scores,
         float score_thres, float nms_thres, std::vector<int>& indices);

void scale_box(cv::Rect& box, cv::Size size);
void scale_box(cv::RotatedRect& rbox, cv::Size size);

void draw_result(cv::Mat& image, const std::string& label,
                 const cv::RotatedRect& rbox);

void post_process(cv::Mat& origin, cv::Mat& result,
                  std::vector<Ort::Value>& outputs);

void main_run(const std::string& imagePath, const std::wstring& modelPath,
              const std::string& outputPath);
