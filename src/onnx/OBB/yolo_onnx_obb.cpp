#include "yolo_onnx_obb.hpp"

#include <onnxruntime_cxx_api.h>

#include "utils.hpp"
// 输入 / 输出名字
#include <string>
#include <vector>

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
}

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

  // 调试打印
  std::cout << "final input_dims_: " << std::endl;
  for (auto d : input_dims_) std::cout << d << ' ';
  std::cout << '\n';
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

/**************************************************
 * @file    yolo_onnx_obb.cpp
 * @brief   YOLOv8 Nms -> OBB
 * @author  姚
 * @date    2025-06-03
 **************************************************/
void nms(std::vector<cv::RotatedRect>& rboxes, std::vector<float>& scores,
         float score_threshold, float nms_threshold,
         std::vector<int>& indices) {
  struct BoxScore {
    cv::RotatedRect box;
    float score;
    int id;
  };
  std::vector<BoxScore> boxes_scores;

  // 1. 首先过滤低于阈值的框
  for (size_t i = 0; i < rboxes.size(); i++) {
    if (scores[i] > score_threshold) {
      boxes_scores.push_back({rboxes[i], scores[i], (int)i});
    }
  }

  // 2. 按照分数排序
  std::sort(
      boxes_scores.begin(), boxes_scores.end(),
      [](const BoxScore& a, const BoxScore& b) { return a.score > b.score; });

  std::vector<bool> isSuppressed(boxes_scores.size(), false);

  // 3. NMS
  for (size_t i = 0; i < boxes_scores.size(); ++i) {
    if (isSuppressed[i]) continue;

    for (size_t j = i + 1; j < boxes_scores.size(); ++j) {
      if (isSuppressed[j]) continue;

      // 计算两个旋转框的IoU
      // 计算两个旋转矩形的交集面积
      cv::Mat intersection_mat;
      float intersection = 0.0f;
      int ret = cv::rotatedRectangleIntersection(
          boxes_scores[i].box, boxes_scores[j].box, intersection_mat);
      if (ret != cv::INTERSECT_NONE) {
        intersection = cv::contourArea(intersection_mat);
      }
      float area1 =
          boxes_scores[i].box.size.width * boxes_scores[i].box.size.height;
      float area2 =
          boxes_scores[j].box.size.width * boxes_scores[j].box.size.height;
      float iou = intersection / (area1 + area2 - intersection);

      if (iou >= nms_threshold) {
        isSuppressed[j] = true;
      }
    }
  }

  // 4. 收集未被抑制的框的索引
  for (size_t i = 0; i < boxes_scores.size(); ++i) {
    if (!isSuppressed[i]) {
      indices.push_back(boxes_scores[i].id);
    }
  }
}

/**************************************************
 * @file    yolo_onnx_obb.cpp
 * @brief   YOLOv8 scale_box->Rect 用于nms，RotatedRect用于 可视化
 * @author  姚
 * @date    2025-06-03
 **************************************************/
void scale_box(cv::Rect& box, cv::Size s) {
  float g =
      std::min(input_width * 1.f / s.width, input_height * 1.f / s.height);
  int pad_w = static_cast<int>((input_width - s.width * g) / 2);
  int pad_h = static_cast<int>((input_height - s.height * g) / 2);

  box.x = static_cast<int>((box.x - pad_w) / g);
  box.y = static_cast<int>((box.y - pad_h) / g);
  box.width = static_cast<int>(box.width / g);
  box.height = static_cast<int>(box.height / g);
}

void scale_box(cv::RotatedRect& rb, cv::Size s) {
  float g =
      std::min(input_width * 1.f / s.width, input_height * 1.f / s.height);
  int pad_w = static_cast<int>((input_width - s.width * g) / 2);
  int pad_h = static_cast<int>((input_height - s.height * g) / 2);

  rb.center -= cv::Point2f(pad_w, pad_h);
  rb.center /= g;
  rb.size /= g;
}
/**************************************************
 * @file    yolo_onnx_obb.cpp
 * @brief   YOLOv8 draw_result 可视化函数
 * @author  姚
 * @date    2025-06-03
 **************************************************/

void draw_result(cv::Mat& img, const std::string& label,
                 const cv::RotatedRect& rb) {
  cv::Point2f pts[4];
  rb.points(pts);
  for (int i = 0; i < 4; ++i)
    cv::line(img, pts[i], pts[(i + 1) % 4], cv::Scalar(255, 0, 0), 1);

  cv::putText(img, label, rb.center, cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(0, 0, 255), 2);
}

/**************************************************
 * @file    yolo_onnx_obb.cpp
 * @brief   YOLOv8 post_process 后处理
 * @author  姚
 * @date    2025-06-03
 **************************************************/
void post_process(cv::Mat& origin, cv::Mat& result,
                  std::vector<Ort::Value>& outputs) {
  if (outputs.empty() || !outputs[0].IsTensor()) {
    std::cout << "[Error] outputs[0] is NOT a tensor!\n";
    return;
  }

  const float* out = outputs[0].GetTensorData<float>();

  std::vector<cv::Rect> boxes;
  std::vector<cv::RotatedRect> rboxes;
  std::vector<float> scores;
  std::vector<int> cls_ids;

  for (int i = 0; i < output_numbox; ++i) {
    // 取分类分数
    const float* class_ptr = out + (4 + i) + 0;  // 指向本 box 的分类起始？
    std::vector<float> cls(num_classes);
    for (int j = 0; j < num_classes; ++j)
      cls[j] = *(out + (4 + j) * output_numbox + i);

    int cid = int(std::max_element(cls.begin(), cls.end()) - cls.begin());
    float score = cls[cid];
    if (score < score_threshold) continue;

    float x = *(out + 0 * output_numbox + i);
    float y = *(out + 1 * output_numbox + i);
    float w = *(out + 2 * output_numbox + i);
    float h = *(out + 3 * output_numbox + i);
    float a = *(out + (4 + num_classes) * output_numbox + i);

    cv::Rect box(int(x - 0.5f * w), int(y - 0.5f * h), int(w), int(h));
    scale_box(box, origin.size());
    boxes.push_back(box);
    scores.push_back(score);
    cls_ids.push_back(cid);

    cv::RotatedRect rb({x, y}, {w, h}, a * 180.f / CV_PI);
    scale_box(rb, origin.size());
    rboxes.push_back(rb);
  }

  std::vector<int> keep;

  nms(rboxes, scores, score_threshold, nms_threshold, keep);

  for (int idx : keep) {
    std::string label =
        class_names[cls_ids[idx]] + ":" + cv::format("%.2f", scores[idx]);
    draw_result(result, label, rboxes[idx]);
  }
}

/**************************************************
 * @file    yolo_onnx_obb.cpp
 * @brief   YOLOv8 main_run 执行函数
 * @author  姚
 * @date    2025-06-03
 **************************************************/

void main_run(const std::string& imagePath, const std::wstring& modelPath,
              const std::string& outputPath) {
  cv::Mat image = cv::imread(imagePath);
  if (image.empty()) {
    std::cerr << "Failed to load image: " << imagePath << std::endl;
  }

  std::vector<float> inputs;
  pre_process(image, inputs);

  std::cout << "  pre_process sucessful " << std::endl;
  std::vector<Ort::Value> outputs;
  process(modelPath.c_str(), inputs, outputs);

  std::cout << "  process sucessful " << std::endl;
  cv::Mat result = image.clone();
  post_process(image, result, outputs);
  std::cout << "  post_process sucessful " << std::endl;

  cv::imwrite(outputPath, result);
}
