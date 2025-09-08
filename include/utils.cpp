#include "utils.hpp"

#include <fstream>
#include <iostream>

// #
/* ===========================================
 * Public
 * To :  LetterBox + pre_process
 * =========================================== */

/**************************************************
 * @file    utils.cpp
 * @brief   YOLOv8 LetterBox
 * @author  姚
 * @date    2025-06-03
 **************************************************/
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
               const cv::Size& newShape, const cv::Scalar& color) {
  cv::Size shape = image.size();
  float r = std::min((float)newShape.height / (float)shape.height,
                     (float)newShape.width / (float)shape.width);
  float ratio[2]{r, r};
  int new_un_pad[2] = {(int)std::round((float)shape.width * r),
                       (int)std::round((float)shape.height * r)};

  auto dw = (float)(newShape.width - new_un_pad[0]) / 2;
  auto dh = (float)(newShape.height - new_un_pad[1]) / 2;

  if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
    cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
  else
    outImage = image.clone();

  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));

  cv::Vec4d params;
  params[0] = ratio[0];
  params[1] = ratio[1];
  params[2] = left;
  params[3] = top;

  cv::copyMakeBorder(outImage, outImage, top, bottom, left, right,
                     cv::BORDER_CONSTANT, color);
  //   cv::imwrite("letterbox.jpg", outImage);
}

/**************************************************
 * @file    utils.cpp
 * @brief   YOLOv8 Seg_LetterBox
 * @author  姚
 * @date    2025-06-03
 **************************************************/

cv::Vec4d Seg_LetterBox(const cv::Mat& image, cv::Mat& outImage,
                        const cv::Size& newShape, const cv::Scalar& color) {
  cv::Size shape = image.size();
  float r = std::min((float)newShape.height / (float)shape.height,
                     (float)newShape.width / (float)shape.width);
  int new_un_pad_w = (int)std::round(shape.width * r);
  int new_un_pad_h = (int)std::round(shape.height * r);

  float dw = (newShape.width - new_un_pad_w) / 2.0f;
  float dh = (newShape.height - new_un_pad_h) / 2.0f;

  if (shape.width != new_un_pad_w || shape.height != new_un_pad_h)
    cv::resize(image, outImage, cv::Size(new_un_pad_w, new_un_pad_h));
  else
    outImage = image.clone();

  cv::copyMakeBorder(outImage, outImage, int(dh), int(dh), int(dw), int(dw),
                     cv::BORDER_CONSTANT, color);
  //   cv::imwrite("letterbox_seg.jpg", outImage);
  return cv::Vec4d(r, r, dw, dh);  // 返回缩放比例r和偏移量dw,dh
}

/**************************************************
 * @file    utils.cpp
 * @brief   YOLOv8 pre_process
 * @author  姚
 * @date    2025-06-03
 **************************************************/

cv::Vec4d Seg_pre_process(cv::Mat& image, std::vector<float>& inputs) {
  cv::Mat seg_letterbox;
  cv::Vec4d params = Seg_LetterBox(image, seg_letterbox, cv::Size(1280, 1280),
                                   cv::Scalar(114, 114, 114));

  cv::cvtColor(seg_letterbox, seg_letterbox, cv::COLOR_BGR2RGB);
  seg_letterbox.convertTo(seg_letterbox, CV_32FC3, 1.0f / 255.0f);
  std::vector<cv::Mat> split_images;
  cv::split(seg_letterbox, split_images);
  for (size_t i = 0; i < seg_letterbox.channels(); ++i) {
    std::vector<float> split_image_data = split_images[i].reshape(1, 1);
    inputs.insert(inputs.end(), split_image_data.begin(),
                  split_image_data.end());
  }
  //   // 将inputs写入文件
  //   FILE* f = fopen("inputs_cpp.bin", "wb");
  //   fwrite(inputs.data(), sizeof(float), inputs.size(), f);
  //   fclose(f);
  return params;
}

/**************************************************
 * @file    utils.cpp
 * @brief   YOLOv8 pre_process
 * @author  姚
 * @date    2025-06-03
 **************************************************/
void pre_process(cv::Mat& image, std::vector<float>& inputs) {
  cv::Vec4d params;
  cv::Mat letterbox;
  LetterBox(image, letterbox, cv::Size(1280, 1280));

  cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);
  letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);
  std::vector<cv::Mat> split_images;
  cv::split(letterbox, split_images);
  for (size_t i = 0; i < letterbox.channels(); ++i) {
    std::vector<float> split_image_data = split_images[i].reshape(1, 1);
    inputs.insert(inputs.end(), split_image_data.begin(),
                  split_image_data.end());
  }
}

/**************************************************
 * @file    yolo_onnx_obb.cpp
 * @brief   YOLOv8 OrtSessionWrapper -> 创建环境等操作
 * @author  姚
 * @date    2025-06-06
 **************************************************/
Ort::Env& OrtSessionWrapper::get_env() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo-obb");
  return env;
}

/**************************************************
 * @file    yolo_onnx_obb.cpp
 * @brief   YOLOv8 setup_io_info -> 获取输入输出信息
 * @author  姚
 * @date    2025-06-06
 **************************************************/

void OrtSessionWrapper::setup_io_info() {
  Ort::AllocatorWithDefaultOptions allocator;

  // Ort::Session 没有 GetInputName 方法，使用 GetInputNameAllocated 替代
  std::cout << "input count: " << session_->GetInputCount() << std::endl;
  for (size_t i = 0; i < session_->GetInputCount(); ++i) {
    Ort::AllocatedStringPtr n = session_->GetInputNameAllocated(i, allocator);
    input_names_.emplace_back(n.get());
  }
  for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
    Ort::AllocatedStringPtr n = session_->GetOutputNameAllocated(i, allocator);
    output_names_.emplace_back(n.get());
  }
  for (auto& s : input_names_) input_names_ptrs_.push_back(s.c_str());
  for (auto& s : output_names_) output_names_ptrs_.push_back(s.c_str());

  // 输入 shape
  Ort::TypeInfo ti = session_->GetInputTypeInfo(0);
  auto tsi = ti.GetTensorTypeAndShapeInfo();
  input_dims_ = tsi.GetShape();  // 可能含 -1

  // 补全未知维度
  if (input_dims_.size() == 4) {  // 默认 NCHW
    if (input_dims_[0] <= 0) input_dims_[0] = 1;
    if (input_dims_[1] <= 0) input_dims_[1] = 3;
    if (input_dims_[2] <= 0) input_dims_[2] = input_height;
    if (input_dims_[3] <= 0) input_dims_[3] = input_width;
  }

  //   // 调试打印
  //   std::cout << "final input_dims_: " << std::endl;
  //   for (auto d : input_dims_) std::cout << d << ' ';
  //   std::cout << '\n';
}

/**************************************************
 * @file    yolo_onnx_obb.cpp
 * @brief   YOLOv8 OrtSessionWrapper -> 构造函数
 * @author  姚
 * @date    2025-06-06
 **************************************************/
OrtSessionWrapper::OrtSessionWrapper(const wchar_t* model_path) {
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
  // 如需 CUDA / TensorRT，加上对应 EP
  OrtCUDAProviderOptions cuda_option;
  cuda_option.device_id = 0;
  cuda_option.arena_extend_strategy = 0;
  cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  cuda_option.gpu_mem_limit = SIZE_MAX;
  cuda_option.do_copy_in_default_stream = 1;
  opts.AppendExecutionProvider_CUDA(cuda_option);
  session_ = std::make_unique<Ort::Session>(get_env(), model_path, opts);
  setup_io_info();
}

/**************************************************
 * @file    yolo_onnx_obb.cpp
 * @brief   YOLOv8 run_inference -> 推理
 * @author  姚
 * @date    2025-06-06
 **************************************************/

void OrtSessionWrapper::run_inference(const std::vector<float>& input_data,
                                      std::vector<Ort::Value>& outputs) {
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem_info, const_cast<float*>(input_data.data()), input_data.size(),
      input_dims_.data(), input_dims_.size());

  outputs = session_->Run(Ort::RunOptions{nullptr},
                          input_names_ptrs_.data(),  // 输入名
                          &input_tensor, 1, output_names_ptrs_.data(),
                          output_names_ptrs_.size());

  //   float* p0 = outputs[0].GetTensorMutableData<float>();
  //   int64_t size0 = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

  //   float* p1 = outputs[1].GetTensorMutableData<float>();
  //   int64_t size1 = outputs[1].GetTensorTypeAndShapeInfo().GetElementCount();

  // 写入二进制文件
  //   FILE* f = fopen("output_cpp.bin", "wb");
  //   fwrite(p0, sizeof(float), size0, f);
  //   fwrite(p1, sizeof(float), size1, f);  // 确保这一行存在！
  //   fclose(f);
}

/**************************************************
 * @file    yolo_onnx_obb.cpp
 * @brief   YOLOv8 process -> Ort run
 * @author  姚
 * @date    2025-06-03
 **************************************************/
void process(const wchar_t* model, std::vector<float>& inputs,
             std::vector<Ort::Value>& outputs) {
  try {
    OrtSessionWrapper session(model);  // 这里如果模型加载失败就会抛
    session.run_inference(inputs, outputs);
    std::cout << "Inference successful!\n";
  } catch (const Ort::Exception& e) {
    std::cerr << "\n=== ORT EXCEPTION ===\n"
              << "ErrorCode : " << e.GetOrtErrorCode() << '\n'
              << "Message   : " << e.what() << "\n\n";
    throw;  // 继续 rethrow 让 VS 保持断点
  }
}

// #
/* ===========================================
 * HBB
 * To :
 * =========================================== */

// #
/* ===========================================
 * OBB
 * To :
 * =========================================== */

// #
/* ===========================================
 * Seg
 * To :
 * =========================================== */