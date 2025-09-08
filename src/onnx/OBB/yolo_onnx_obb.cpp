#include "yolo_onnx_obb.hpp"

#include <onnxruntime_cxx_api.h>

#include "utils.hpp"
// 输入 / 输出名字
#include <iostream>
#include <string>
#include <vector>

/**************************************************
 * @file    yolo_onnx_obb.cpp
 * @brief   YOLOv8 Nms -> OBB
 * @author  姚
 * @date    2025-06-03
 **************************************************/
float computeIoU(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
  std::vector<cv::Point2f> intersectionPts;
  auto result = cv::rotatedRectangleIntersection(rect1, rect2, intersectionPts);

  if (result == cv::INTERSECT_NONE || intersectionPts.empty()) {
    return 0.0f;
  }

  std::vector<cv::Point2f> hull;
  cv::convexHull(intersectionPts, hull);
  float intersectionArea = static_cast<float>(cv::contourArea(hull));

  float area1 = rect1.size.width * rect1.size.height;
  float area2 = rect2.size.width * rect2.size.height;
  float unionArea = area1 + area2 - intersectionArea;

  return intersectionArea / unionArea;
}

/**************************************************
 * @file    yolo_onnx_obb.cpp
 * @brief   YOLOv8 Nms -> OBB
 * @author  姚
 * @date    2025-06-03
 **************************************************/
void obb_nms(std::vector<cv::RotatedRect>& rboxes, std::vector<float>& scores,
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
      float iou = computeIoU(boxes_scores[i].box, boxes_scores[j].box);

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
  auto det_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

  int output_numbox = det_shape[1];  // 33600
  int det_feat_dim = det_shape[2];   // 7
  int num_class = det_shape[2] - 4;  // 类别数
  std::cout << "  output_numbox  " << output_numbox << std::endl;
  std::cout << "  det_feat_dim  " << det_feat_dim << std::endl;

  std::vector<cv::RotatedRect> rboxes;
  std::vector<float> scores;
  std::vector<int> cls_ids;

  for (int i = 0; i < output_numbox; ++i) {
    const float* ptr = out + i * det_feat_dim;

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
    // std::cout << "[Debug] max_conf: " << max_conf << std::endl;

    float cx = ptr[0];
    float cy = ptr[1];
    float w = ptr[2];
    float h = ptr[3];
    float a = ptr[4 + num_class];

    cv::RotatedRect rb({cx, cy}, {w, h}, a * 180.f / CV_PI);
    scale_box(rb, origin.size());
    scores.push_back(max_conf);
    cls_ids.push_back(label);
    rboxes.push_back(rb);
  }
  std::vector<int> keep;

  std::cout << "  rboxes.size()  " << rboxes.size() << std::endl;
  obb_nms(rboxes, scores, score_threshold, nms_threshold, keep);
  std::cout << "  keep.size()  " << keep.size() << std::endl;

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
