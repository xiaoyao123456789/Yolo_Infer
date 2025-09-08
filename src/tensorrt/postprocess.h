#pragma once

#include <cuda_runtime_api.h>

#include <string>
#include <vector>

// TensorRT条件编译包含
#ifdef TENSORRT_AVAILABLE
#include "NvInfer.h"
#endif

// ============================================================================
// 后处理函数声明
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CUDA解码函数
 *
 * 将YOLO网络的原始预测输出解码为可用的检测结果。
 * 该函数在GPU上并行处理多个预测框，进行坐标转换、置信度筛选和多类别处理。
 *
 * @param predict 网络预测输出数组指针（设备内存）
 * @param num_bboxes 预测边界框的数量
 * @param confidence_threshold 置信度阈值，低于此值的检测框将被过滤
 * @param parray 输出检测结果数组指针（设备内存）
 *               格式：[count, bbox1, bbox2, ...]
 *               每个bbox格式：[x1, y1, x2, y2, confidence, class_id, keep_flag]
 * @param max_objects 最大输出对象数量限制
 * @param stream CUDA流，用于异步执行
 * @param box_element 每个边界框的元素数量（默认7：x1,y1,x2,y2,conf,class,keep）
 * @param num_class 类别数量（默认80，对应COCO数据集）
 */
void cuda_decode(float* predict, int num_bboxes, float confidence_threshold,
                 float* parray, int max_objects, cudaStream_t stream,
                 int box_element = 5, int num_class = 1);

/**
 * @brief CUDA非极大值抑制（NMS）函数
 *
 * 对解码后的检测结果执行非极大值抑制，去除重复的检测框。
 * 对于每个类别，保留置信度最高的检测框，抑制与其IoU超过阈值的其他框。
 *
 * @param parray 检测结果数组指针（设备内存）
 *               输入格式：[count, bbox1, bbox2, ...]
 *               每个bbox格式：[x1, y1, x2, y2, confidence, class_id, keep_flag]
 *               输出：keep_flag被更新，0表示被抑制，1表示保留
 * @param nms_threshold NMS阈值，通常设置为0.5
 *                      当两个同类别框的IoU大于此值时，置信度低的框被抑制
 * @param max_objects 最大对象数量，与解码时的设置保持一致
 * @param stream CUDA流，用于异步执行
 */
void cuda_nms(float* parray, float nms_threshold, int max_objects,
              cudaStream_t stream);

/**
 * @brief CPU解码函数
 *
 * CPU版本的YOLO解码函数，处理(5, 8400)格式的网络输出
 *
 * @param predict 网络预测输出数组指针（主机内存）
 * @param num_bboxes 预测边界框的数量
 * @param confidence_threshold 置信度阈值
 * @param parray 输出检测结果数组指针（主机内存）
 * @param max_objects 最大输出对象数量限制
 * @param box_element 每个边界框的元素数量
 * @param num_class 类别数量
 */
void cpu_decode(float* predict, int num_bboxes, float confidence_threshold,
                float* parray, int max_objects, int box_element = 5,
                int num_class = 1);

/**
 * @brief CPU非极大值抑制（NMS）函数
 *
 * CPU版本的NMS函数
 *
 * @param parray 检测结果数组指针（主机内存）
 * @param nms_threshold NMS阈值
 * @param max_objects 最大对象数量
 */
void cpu_nms(float* parray, float nms_threshold, int max_objects);

#ifdef __cplusplus
}
#endif