#include "yolo_onnx_seg.hpp"

#include <iostream>

#include "utils.hpp"

/**************************************************
 * @file    yolo_onnx_seg.cpp
 * @brief   YOLOv8 nms 非极大值抑制
 * @author  姚
 * @date    2025-06-03
 **************************************************/
void seg_nms(const std::vector<cv::Rect>& boxes,
             const std::vector<float>& scores, float score_threshold,
             float nms_threshold, std::vector<int>& indices) {
  struct BoxScore {
    cv::Rect box;
    float score;
    int id;
  };
  std::vector<BoxScore> boxes_scores;

  // 1. 首先过滤低于阈值的框
  for (size_t i = 0; i < boxes.size(); i++) {
    boxes_scores.push_back({boxes[i], scores[i], (int)i});
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

      // 计算两个水平框的IoU
      const auto& box1 = boxes_scores[i].box;
      const auto& box2 = boxes_scores[j].box;

      // 计算交集区域
      int x1 = std::max(box1.x, box2.x);
      int y1 = std::max(box1.y, box2.y);
      int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
      int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

      // 如果没有交集，跳过
      if (x2 <= x1 || y2 <= y1) continue;

      // 计算交集面积
      float intersection = (x2 - x1) * (y2 - y1);
      // 计算两个框的面积
      float area1 = box1.width * box1.height;
      float area2 = box2.width * box2.height;
      // 计算IoU
      float iou = intersection / (area1 + area2 - intersection);

      if (iou >= nms_threshold) {
        isSuppressed[j] = true;
      }
    }
  }

  for (size_t i = 0; i < boxes_scores.size(); ++i) {
    if (!isSuppressed[i]) {
      indices.push_back(boxes_scores[i].id);
    }
  }
}

/**************************************************
 * @file    yolo_onnx_seg.cpp
 * @brief   YOLOv8 draw_result 可视化函数
 * @author  姚
 * @date    2025-06-03
 **************************************************/
void seg_draw_result(cv::Mat& img, const std::string& label,
                     const cv::Rect& box, const cv::Mat& mask) {
  // 选颜色
  int color_idx = 0;  // 默认颜色
  // 这里可拓展传入cls_id索引，简化演示用0
  cv::Scalar color = COLORS[color_idx % NUM_COLORS];

  // 绘制半透明掩码区域
  if (!mask.empty() && mask.type() == CV_32F && mask.size() == img.size()) {
    cv::Mat colored_mask;
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) channels[i] = mask * color[i];
    cv::merge(channels, colored_mask);

    // 叠加：img = img * (1 - alpha * mask) + colored_mask * alpha * mask
    double alpha = 0.5;
    for (int y = 0; y < img.rows; ++y) {
      uchar* img_ptr = img.ptr<uchar>(y);
      const float* mask_ptr = mask.ptr<float>(y);
      const cv::Vec3b* color_ptr = colored_mask.ptr<cv::Vec3b>(y);
      for (int x = 0; x < img.cols; ++x) {
        float m = mask_ptr[x];
        if (m > 0) {
          for (int c = 0; c < 3; ++c) {
            img_ptr[3 * x + c] = uchar(img_ptr[3 * x + c] * (1 - alpha * m) +
                                       color_ptr[x][c] * alpha * m);
          }
        }
      }
    }
  }

  // 绘制边框
  cv::rectangle(img, box, color, 2);

  // 绘制文字背景框和文字
  int baseLine = 0;
  cv::Size label_size =
      cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  int top = std::max(box.y, label_size.height);
  cv::rectangle(img, cv::Point(box.x, top - label_size.height - baseLine),
                cv::Point(box.x + label_size.width, top + baseLine), color,
                cv::FILLED);
  cv::putText(img, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 0, 0), 1);
}

/**************************************************
 * @file    yolo_onnx_seg.cpp
 * @brief   YOLOv8 scale_box-> 缩放框
 * @author  姚
 * @date    2025-06-03
 **************************************************/
void seg_scale_box(cv::Rect& box, cv::Size origin_size, float r, float dw,
                   float dh) {
  //   std::cout << "LetterBox r=" << r << " dw=" << dw << " dh=" << dh <<
  //   std::endl;

  int x = int((box.x - dw) / r);
  int y = int((box.y - dh) / r);
  int w = int(box.width / r);
  int h = int(box.height / r);

  x = std::max(0, x);
  y = std::max(0, y);
  if (x + w > origin_size.width) w = origin_size.width - x;
  if (y + h > origin_size.height) h = origin_size.height - y;
  if (w < 0) w = 0;
  if (h < 0) h = 0;

  box = cv::Rect(x, y, w, h);
}

/**************************************************
 * @file    yolo_onnx_seg.cpp
 * @brief   YOLOv8 post_process_seg 后处理
 * @author  姚
 * @date    2025-06-03
 **************************************************/
void post_process_seg(cv::Mat& origin, cv::Mat& result,
                      std::vector<Ort::Value>& outputs,
                      float r,   // r
                      float dw,  // dw
                      float dh) {
  if (outputs.empty() || !outputs[0].IsTensor() || !outputs[1].IsTensor()) {
    std::cout << "[Error] outputs are NOT tensors!\n";
    return;
  }

  const float* det_out = outputs[0].GetTensorData<float>();  // 检测输出
  const float* seg_out = outputs[1].GetTensorData<float>();  // 分割输出

  // 获取输出维度信息
  auto det_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  auto seg_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();

  int output_numboxs = det_shape[1];                  // 33600
  int det_feat_dim = det_shape[2];                    // 37
  int num_class = det_shape[2] - seg_shape[1] - 4;    // 类别数
  int num_prototypes = det_feat_dim - 4 - num_class;  // TODO 修改此处
  int proto_h = seg_shape[2];                         // 原型高度
  int proto_w = seg_shape[3];                         // 原型宽度

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> cls_ids;
  std::vector<std::vector<float>> seg_coeffs;  // 存储分割系数

  std::cout << "[Info] det_shape: " << det_shape[0] << "  " << det_shape[1]
            << "  " << det_shape[2] << std::endl;

  std::cout << "[Info] num_class:" << num_class << std::endl;

  for (int i = 0; i < output_numboxs; ++i) {
    const float* ptr = det_out + i * det_feat_dim;

    // 找到最大类别的索引和得分
    int label = 0;
    float max_conf = ptr[4];  // 第一个类分数
    for (int c = 1; c < num_class; ++c) {
      float conf = ptr[4 + c];
      if (conf > max_conf) {
        max_conf = conf;
        label = c;
      }
    }

    if (max_conf < score_threshold) continue;

    // 解码 bbox
    float cx = ptr[0];
    float cy = ptr[1];
    float w = ptr[2];
    float h = ptr[3];
    int left = static_cast<int>(cx - 0.5f * w);
    int top = static_cast<int>(cy - 0.5f * h);
    int width = static_cast<int>(w);
    int height = static_cast<int>(h);
    cv::Rect box(left, top, width, height);

    // 解析 mask 系数
    std::vector<float> coeffs;
    for (int j = 0; j < num_prototypes; ++j) {
      coeffs.push_back(ptr[4 + num_class + j]);  // 紧跟在类别之后
    }

    // 收集
    boxes.push_back(box);
    scores.push_back(max_conf);
    cls_ids.push_back(label);
    seg_coeffs.push_back(coeffs);
  }

  std::cout << "boxes.size(): " << boxes.size() << std::endl;
  std::cout << "scores.size(): " << scores.size() << std::endl;
  std::cout << "cls_ids.size(): " << cls_ids.size() << std::endl;
  std::cout << "seg_coeffs.size(): " << seg_coeffs.size() << std::endl;

  // NMS
  std::vector<int> keep;
  seg_nms(boxes, scores, score_threshold, nms_threshold, keep);

  std::cout << "[Debug] keep.size(): " << keep.size() << std::endl;

  //   // 生成最终掩码
  for (int idx : keep) {
    cv::Rect scaled_box = boxes[idx];

    // std::cout << "[Info] scaled_box qian: " << scaled_box <<
    //     std::endl;
    seg_scale_box(scaled_box, origin.size(), r, dw, dh);
    // std::cout << "[Info] scaled_box hou: " << scaled_box << std::endl;

    cv::Mat mask = cv::Mat::zeros(proto_h, proto_w, CV_32F);
    // 用分割系数和原型叠加mask
    for (int i = 0; i < proto_h; ++i) {
      for (int j = 0; j < proto_w; ++j) {
        float pixel_val = 0.0f;
        for (int k = 0; k < num_prototypes; ++k) {
          pixel_val += seg_coeffs[idx][k] *
                       seg_out[k * proto_h * proto_w + i * proto_w + j];
        }
        mask.at<float>(i, j) = pixel_val;
      }
    }
    // sigmoid激活
    cv::Mat mask_sigmoid;
    cv::exp(-mask, mask_sigmoid);
    mask_sigmoid = 1.0 / (1.0 + mask_sigmoid);

    // 缩放到原图尺寸
    cv::Mat final_mask;
    cv::resize(mask_sigmoid, final_mask, origin.size(), 0, 0, cv::INTER_LINEAR);

    // 裁剪掩码到目标框
    cv::Mat cropped_mask = cv::Mat::zeros(origin.size(), CV_32F);

    if (scaled_box.width > 0 && scaled_box.height > 0 && scaled_box.x >= 0 &&
        scaled_box.y >= 0 && scaled_box.x + scaled_box.width <= origin.cols &&
        scaled_box.y + scaled_box.height <= origin.rows) {
      cv::Mat roi_mask = final_mask(scaled_box);
      if (!roi_mask.empty() &&
          roi_mask.size() == cv::Size(scaled_box.width, scaled_box.height)) {
        roi_mask.copyTo(cropped_mask(scaled_box));
      }
    }
    // cv::imwrite("debug_mask.png", mask * 255);

    std::string label =
        class_names[cls_ids[idx]] + ":" + cv::format("%.2f", scores[idx]);

    seg_draw_result(result, label, scaled_box, cropped_mask);
  }
}

/**************************************************
 * @file    yolo_onnx_seg.cpp
 * @brief   YOLOv8 main_run 执行函数
 * @author  姚
 * @date    2025-06-03
 **************************************************/
void main_seg_run(const std::string& imagePath, const std::wstring& modelPath,
                  const std::string& outputPath) {
  cv::Mat image = cv::imread(imagePath);
  if (image.empty()) {
    std::cerr << "Failed to load image: " << imagePath << std::endl;
  }

  std::vector<float> inputs;
  //   std::cout << "inputs1.size(): " << inputs.size() << "\n";

  cv::Vec4d letterbox_params = Seg_pre_process(image, inputs);

  std::vector<Ort::Value> outputs;
  //   std::cout << "inputs2.size(): " << inputs.size() << "\n";
  process(modelPath.c_str(), inputs, outputs);
  //   std::cout << "output1.size(): " << outputs.size() << "\n";
  cv::Mat result = image.clone();

  post_process_seg(image, result, outputs, (float)letterbox_params[0],  // r
                   (float)letterbox_params[2],                          // dw
                   (float)letterbox_params[3]);

  cv::imwrite(outputPath, result);
}