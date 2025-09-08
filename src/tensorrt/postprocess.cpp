#include "postprocess.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>


/**
 * @brief CPU版本的YOLO后处理函数
 *
 * 模仿Python版本的解码逻辑，处理(5, 8400)格式的网络输出
 *
 * @param predict 网络预测输出数组 (5 * 8400)
 * @param num_bboxes 预测框数量 (8400)
 * @param confidence_threshold 置信度阈值
 * @param parray 输出数组
 * @param max_objects 最大对象数量
 * @param box_element 每个框的元素数量
 * @param num_class 类别数量
 */
void cpu_decode(float* predict, int num_bboxes, float confidence_threshold,
                float* parray, int max_objects, int box_element,
                int num_class) {
  std::cout << "[CPU Decode] 开始CPU解码，预测框数量: " << num_bboxes
            << std::endl;
  std::cout << "[CPU Decode] 置信度阈值: " << confidence_threshold << std::endl;

  // 初始化输出数组计数器
  parray[0] = 0;
  int valid_count = 0;

  // 遍历所有预测框
  for (int i = 0; i < num_bboxes; i++) {
    // 按照Python版本的数据布局访问：
    // predict[0*8400 + i] = cx
    // predict[1*8400 + i] = cy
    // predict[2*8400 + i] = w
    // predict[3*8400 + i] = h
    // predict[4*8400 + i] = conf

    float cx = predict[0 * num_bboxes + i];          // 中心x坐标
    float cy = predict[1 * num_bboxes + i];          // 中心y坐标
    float w = predict[2 * num_bboxes + i];           // 宽度
    float h = predict[3 * num_bboxes + i];           // 高度
    float confidence = predict[4 * num_bboxes + i];  // 置信度

    // 置信度阈值筛选
    if (confidence < confidence_threshold) {
      continue;
    }

    // 检查是否超出最大对象数量
    if (valid_count >= max_objects) {
      break;
    }

    // 转换为左上角-右下角格式
    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;
    float x2 = cx + w * 0.5f;
    float y2 = cy + h * 0.5f;

    // 写入输出数组
    int base_idx = 1 + valid_count * box_element;
    parray[base_idx + 0] = x1;          // 左上角x坐标
    parray[base_idx + 1] = y1;          // 左上角y坐标
    parray[base_idx + 2] = x2;          // 右下角x坐标
    parray[base_idx + 3] = y2;          // 右下角y坐标
    parray[base_idx + 4] = confidence;  // 置信度
    parray[base_idx + 5] = 0;           // 类别标签（默认0）
    parray[base_idx + 6] = 1;           // keep_flag（初始设为1）

    valid_count++;
  }

  // 更新输出数组的计数器
  parray[0] = valid_count;

  std::cout << "[CPU Decode] 解码完成，有效检测框数量: " << valid_count
            << std::endl;
}

/**
 * @brief CPU版本的NMS函数
 *
 * @param parray 检测结果数组
 * @param nms_threshold NMS阈值
 * @param max_objects 最大对象数量
 */
void cpu_nms(float* parray, float nms_threshold, int max_objects) {
  int count = static_cast<int>(parray[0]);
  std::cout << "[CPU NMS] 开始NMS，输入检测框数量: " << count << std::endl;
  std::cout << "[CPU NMS] NMS阈值: " << nms_threshold << std::endl;

  const int bbox_element = 7;  // [x1, y1, x2, y2, conf, class, keep_flag]

  // 按置信度排序
  std::vector<int> indices(count);
  for (int i = 0; i < count; i++) {
    indices[i] = i;
  }

  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    float conf_a = parray[1 + a * bbox_element + 4];
    float conf_b = parray[1 + b * bbox_element + 4];
    return conf_a > conf_b;  // 降序排列
  });

  // NMS处理
  for (int i = 0; i < count; i++) {
    int idx_i = indices[i];
    float* box_i = &parray[1 + idx_i * bbox_element];

    // 如果已被抑制，跳过
    if (box_i[6] == 0) continue;

    for (int j = i + 1; j < count; j++) {
      int idx_j = indices[j];
      float* box_j = &parray[1 + idx_j * bbox_element];

      // 如果已被抑制，跳过
      if (box_j[6] == 0) continue;

      // 检查类别是否相同
      if (box_i[5] != box_j[5]) continue;

      // 计算IoU
      float x1_i = box_i[0], y1_i = box_i[1], x2_i = box_i[2], y2_i = box_i[3];
      float x1_j = box_j[0], y1_j = box_j[1], x2_j = box_j[2], y2_j = box_j[3];

      float left = std::max(x1_i, x1_j);
      float top = std::max(y1_i, y1_j);
      float right = std::min(x2_i, x2_j);
      float bottom = std::min(y2_i, y2_j);

      float intersection_area = 0.0f;
      if (right > left && bottom > top) {
        intersection_area = (right - left) * (bottom - top);
      }

      float area_i = (x2_i - x1_i) * (y2_i - y1_i);
      float area_j = (x2_j - x1_j) * (y2_j - y1_j);
      float union_area = area_i + area_j - intersection_area;

      float iou = intersection_area / union_area;

      // 如果IoU超过阈值，抑制置信度较低的框
      if (iou > nms_threshold) {
        box_j[6] = 0;  // 标记为抑制
      }
    }
  }

  // 统计最终保留的检测框数量
  int final_count = 0;
  for (int i = 0; i < count; i++) {
    float* box = &parray[1 + i * bbox_element];
    if (box[6] == 1) {
      final_count++;
    }
  }

  // 更新parray[0]为实际保留的检测框数量
  parray[0] = final_count;

  std::cout << "[CPU NMS] NMS完成，最终保留检测框数量: " << final_count
            << std::endl;
}