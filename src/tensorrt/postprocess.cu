#include <NvInfer.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_utils.hpp"
#include "postprocess.h"
#include "trt_utils.hpp"

// ============================================================================
// 常量定义
// ============================================================================

// ============================================================================
// 函数声明
// ============================================================================

/**
 * @brief 计算两个旋转边界框之间的概率IoU
 * @param ax, ay 第一个框的中心坐标
 * @param aw, ah 第一个框的宽度和高度
 * @param aangle 第一个框的旋转角度
 * @param bx, by 第二个框的中心坐标
 * @param bw, bh 第二个框的宽度和高度
 * @param bangle 第二个框的旋转角度
 * @return 返回IoU值
 */
__device__ float box_probiou(float ax, float ay, float aw, float ah,
                             float aangle, float bx, float by, float bw,
                             float bh, float bangle);

/**
 * @brief 非极大值抑制（NMS）CUDA核函数
 * @param parray 检测结果数组
 * @param max_objects 最大对象数量
 * @param threshold NMS阈值
 */
__global__ void nms_kernel(float* parray, int max_objects, float threshold);

/**
 * @brief 针对旋转边界框的NMS核函数
 * @param bboxes 边界框数组
 * @param max_objects 最大对象数量
 * @param threshold NMS阈值
 */
__global__ void nms_kernel_obb(float* bboxes, int max_objects, float threshold);

/**
 * @brief 解码YOLO预测结果的CUDA核函数
 * @param predict 预测结果数组
 * @param num_bboxes 边界框数量
 * @param confidence_threshold 置信度阈值
 * @param parray 输出数组
 * @param max_objects 最大对象数量
 * @param box_element 每个框的元素数量
 * @param num_class 类别数量
 */
__global__ void decode_kernel(float* predict, int num_bboxes,
                              float confidence_threshold, float* parray,
                              int max_objects, int box_element, int num_class);

// ============================================================================
// 设备函数实现
// ============================================================================

/**
 * @brief 计算两个边界框的概率IoU（交并比）
 *
 * 这是一个简化的IoU计算实现，主要用于水平边界框。
 * 在实际的旋转目标检测项目中，需要更复杂的旋转框IoU计算算法。
 *
 * IoU = 交集面积 / 并集面积
 *
 * @param ax, ay 第一个框的中心坐标
 * @param aw, ah 第一个框的宽度和高度
 * @param aangle 第一个框的旋转角度（当前实现中未使用）
 * @param bx, by 第二个框的中心坐标
 * @param bw, bh 第二个框的宽度和高度
 * @param bangle 第二个框的旋转角度（当前实现中未使用）
 * @return 返回IoU值，范围[0, 1]
 */
__device__ float box_probiou(float ax, float ay, float aw, float ah,
                             float aangle, float bx, float by, float bw,
                             float bh, float bangle) {
  // 计算两个框的边界坐标
  // 第一个框的左边界：中心x坐标减去宽度的一半
  // 第二个框的左边界：中心x坐标减去宽度的一半
  // 取两者的最大值作为交集的左边界
  float left = fmaxf(ax - aw / 2, bx - bw / 2);

  // 计算交集的右边界：取两个框右边界的最小值
  float right = fminf(ax + aw / 2, bx + bw / 2);

  // 计算交集的上边界：取两个框上边界的最大值
  float top = fmaxf(ay - ah / 2, by - bh / 2);

  // 计算交集的下边界：取两个框下边界的最小值
  float bottom = fminf(ay + ah / 2, by + bh / 2);

  // 计算交集的宽度和高度
  // 如果没有交集，宽度或高度为0
  float width = fmaxf(0.0f, right - left);
  float height = fmaxf(0.0f, bottom - top);

  // 计算交集面积
  float intersection = width * height;

  // 计算两个框的面积
  float area_a = aw * ah;
  float area_b = bw * bh;

  // 计算并集面积：两个框面积之和减去交集面积
  float union_area = area_a + area_b - intersection;

  // 返回IoU值，添加小的epsilon值避免除零错误
  return intersection / (union_area + 1e-7f);
}

// ============================================================================
// CUDA核函数实现
// ============================================================================

// 声明外部定义的常量
extern const int bbox_element;

/**
 * @brief 非极大值抑制（NMS）核函数
 *
 * NMS用于去除重复的检测框，保留置信度最高的框。
 * 对于每个检测框，如果存在另一个同类别且置信度更高的框与其IoU超过阈值，
 * 则将当前框标记为抑制状态。
 *
 * @param parray 检测结果数组，格式：[count, bbox1, bbox2, ...]
 *               每个bbox格式：[x1, y1, x2, y2, confidence, class_id, keep_flag]
 * @param max_objects 最大对象数量
 * @param threshold NMS阈值，通常为0.5
 */
__global__ void nms_kernel(float* parray, int max_objects, float threshold) {
  // 计算当前线程处理的检测框索引
  int position = (blockDim.x * blockIdx.x + threadIdx.x);

  // 获取实际检测框数量，不超过最大对象数量
  int count = min((int)parray[0], max_objects);

  // 如果当前线程索引超出检测框数量，直接返回
  if (position >= count) return;

  // 获取当前线程处理的检测框指针
  // parray[0]存储检测框数量，从parray[1]开始存储检测框数据
  float* pcurrent = parray + 1 + position * bbox_element;

  // 遍历所有检测框，进行NMS比较
  for (int i = 0; i < count; ++i) {
    // 获取比较框的指针
    float* pitem = parray + 1 + i * bbox_element;

    // 跳过自己或不同类别的框
    // pcurrent[5]和pitem[5]分别是当前框和比较框的类别ID
    if (i == position || pcurrent[5] != pitem[5]) continue;

    // 如果比较框的置信度大于等于当前框的置信度
    // pcurrent[4]和pitem[4]分别是当前框和比较框的置信度
    if (pitem[4] >= pcurrent[4]) {
      // 如果置信度相等且比较框索引小于当前框索引，跳过
      // 这样可以保证相同置信度的框中索引小的被保留
      if (pitem[4] == pcurrent[4] && i < position) continue;

      // 计算两个框的IoU
      // 将左上角右下角坐标转换为中心坐标和宽高
      float cx1 = (pcurrent[0] + pcurrent[2]) * 0.5f;  // 当前框中心x
      float cy1 = (pcurrent[1] + pcurrent[3]) * 0.5f;  // 当前框中心y
      float w1 = pcurrent[2] - pcurrent[0];            // 当前框宽度
      float h1 = pcurrent[3] - pcurrent[1];            // 当前框高度
      
      float cx2 = (pitem[0] + pitem[2]) * 0.5f;       // 比较框中心x
      float cy2 = (pitem[1] + pitem[3]) * 0.5f;       // 比较框中心y
      float w2 = pitem[2] - pitem[0];                 // 比较框宽度
      float h2 = pitem[3] - pitem[1];                 // 比较框高度
      
      float iou = box_probiou(cx1, cy1, w1, h1, 0, cx2, cy2, w2, h2, 0);

      // 如果IoU超过阈值，标记当前框为抑制状态
      if (iou > threshold) {
        pcurrent[6] = 0;  // 将keep_flag设置为0，表示被抑制
        return;
      }
    }
  }
}

/**
 * @brief 针对旋转边界框（OBB）的NMS核函数
 *
 * 与普通NMS类似，但使用旋转框的IoU计算方法。
 * 旋转框包含角度信息，需要考虑框的旋转角度。
 *
 * @param bboxes 旋转边界框数组
 * @param max_objects 最大对象数量
 * @param threshold NMS阈值
 */
static __global__ void nms_kernel_obb(float* bboxes, int max_objects,
                                      float threshold) {
  // 计算当前线程处理的检测框索引
  int position = (blockDim.x * blockIdx.x + threadIdx.x);

  // 获取检测框数量
  int count = bboxes[0];

  // 边界检查
  if (position >= count) return;

  // 获取当前处理的旋转框指针
  float* pcurrent = bboxes + 1 + position * bbox_element;

  // 遍历所有旋转框进行NMS比较
  for (int i = 0; i < count; ++i) {
    float* pitem = bboxes + 1 + i * bbox_element;

    // 跳过自己或不同类别的框
    if (i == position || pcurrent[5] != pitem[5]) continue;

    // 置信度比较
    if (pitem[4] >= pcurrent[4]) {
      if (pitem[4] == pcurrent[4] && i < position) continue;

      // 计算旋转框IoU
      // 将左上角右下角坐标转换为中心坐标和宽高
      float cx1 = (pcurrent[0] + pcurrent[2]) * 0.5f;  // 当前框中心x
      float cy1 = (pcurrent[1] + pcurrent[3]) * 0.5f;  // 当前框中心y
      float w1 = pcurrent[2] - pcurrent[0];            // 当前框宽度
      float h1 = pcurrent[3] - pcurrent[1];            // 当前框高度
      
      float cx2 = (pitem[0] + pitem[2]) * 0.5f;       // 比较框中心x
      float cy2 = (pitem[1] + pitem[3]) * 0.5f;       // 比较框中心y
      float w2 = pitem[2] - pitem[0];                 // 比较框宽度
      float h2 = pitem[3] - pitem[1];                 // 比较框高度
      
      // pcurrent[7]和pitem[7]分别是当前框和比较框的旋转角度
      float iou = box_probiou(cx1, cy1, w1, h1, pcurrent[7], cx2, cy2, w2, h2, pitem[7]);

      // IoU阈值判断
      if (iou > threshold) {
        pcurrent[6] = 0;  // 标记为抑制状态
        return;
      }
    }
  }
}

/**
 * @brief YOLO预测结果解码核函数
 *
 * 将YOLO网络的原始输出转换为可用的检测结果。
 * 包括坐标转换、置信度筛选、多类别处理等。
 *
 * @param predict 网络预测输出数组
 * @param num_bboxes 预测框数量
 * @param confidence_threshold 置信度阈值
 * @param parray 输出检测结果数组
 * @param max_objects 最大对象数量
 * @param box_element 每个框的元素数量
 * @param num_class 类别数量
 */
static __global__ void decode_kernel(float* predict, int num_bboxes,
                                     float confidence_threshold, float* parray,
                                     int max_objects, int box_element,
                                     int num_class) {
  // 计算当前线程处理的预测框索引
  int position = (blockDim.x * blockIdx.x + threadIdx.x);

  // 边界检查：使用传入的num_bboxes参数而不是predict[0]
  if (position >= num_bboxes) return;
  
  // 调试：检查关键参数
  if (position == 0) {
    printf("[GPU DEBUG] decode_kernel: num_bboxes=%d, max_objects=%d, box_element=%d, num_class=%d\n", 
           num_bboxes, max_objects, box_element, num_class);
  }

  // 按照CPU版本的数据布局访问：predict[channel * num_bboxes + position]
  // 添加边界检查防止数组越界
  int base_idx = position;
  if (base_idx >= num_bboxes) {
    if (position == 0) printf("[GPU ERROR] base_idx=%d >= num_bboxes=%d\n", base_idx, num_bboxes);
    return;
  }
  
  float cx = predict[0 * num_bboxes + base_idx];          // 中心x坐标
  float cy = predict[1 * num_bboxes + base_idx];          // 中心y坐标
  float w = predict[2 * num_bboxes + base_idx];           // 宽度
  float h = predict[3 * num_bboxes + base_idx];           // 高度
  float confidence = predict[4 * num_bboxes + base_idx];  // 置信度
  
  // 调试：检查数据有效性
  if (position < 5) {
    printf("[GPU DEBUG] position=%d: cx=%.3f, cy=%.3f, w=%.3f, h=%.3f, conf=%.3f\n", 
           position, cx, cy, w, h, confidence);
  }
  
  int label = 0;  // 默认类别标签
  float max_conf = confidence;

  // 多类别情况：寻找置信度最高的类别
  if (num_class > 1) {
    for (int c = 1; c < num_class; ++c) {
      float conf = predict[(4 + c) * num_bboxes + position];  // 获取第c类的置信度
      if (conf > max_conf) {
        max_conf = conf;
        label = c;  // 更新最佳类别标签
      }
    }
    confidence = max_conf;  // 使用最高置信度
  }

  // 置信度阈值筛选 - 在原子操作之前进行
  if (confidence < confidence_threshold) return;

  // 原子操作增加输出数组的计数器，获取当前框在输出数组中的索引
  int index = atomicAdd(parray, 1);

  // 如果输出数组已满，直接返回
  if (index >= max_objects) return;

  // 转换为左上角-右下角格式
  float x1 = cx - w * 0.5f;  // 左上角x坐标
  float y1 = cy - h * 0.5f;  // 左上角y坐标
  float x2 = cx + w * 0.5f;  // 右下角x坐标
  float y2 = cy + h * 0.5f;  // 右下角y坐标

  // 将解码结果写入输出数组
  float* pout_item = parray + 1 + index * bbox_element;
  *pout_item++ = x1;          // 左上角x坐标
  *pout_item++ = y1;          // 左上角y坐标
  *pout_item++ = x2;          // 右下角x坐标
  *pout_item++ = y2;          // 右下角y坐标
  *pout_item++ = confidence;  // 置信度
  *pout_item++ = label;       // 类别标签
  *pout_item++ = 1;           // keep_flag，初始设为1（保留）
}

// ============================================================================
// C接口函数
// ============================================================================

// ============================================================================
// C接口函数实现
// ============================================================================

extern "C" {

/**
 * @brief CUDA解码函数的C接口
 *
 * 启动CUDA核函数进行YOLO预测结果解码。
 *
 * @param predict 预测结果数组指针
 * @param num_bboxes 边界框数量
 * @param confidence_threshold 置信度阈值
 * @param parray 输出数组指针
 * @param max_objects 最大对象数量
 * @param stream CUDA流
 * @param box_element 每个框的元素数量
 * @param num_class 类别数量
 */
void cuda_decode(float* predict, int num_bboxes, float confidence_threshold,
                 float* parray, int max_objects, cudaStream_t stream,
                 int box_element, int num_class) {
  // 设置CUDA执行配置
  int block = 256;                             // 每个块的线程数
  int grid = ceil(num_bboxes / (float)block);  // 块的数量

  // 启动解码核函数
  decode_kernel<<<grid, block, 0, stream>>>(
      (float*)predict, num_bboxes, confidence_threshold, parray, max_objects,
      box_element, num_class);
}

/**
 * @brief CUDA NMS函数的C接口
 *
 * 启动CUDA核函数进行非极大值抑制。
 *
 * @param parray 检测结果数组指针
 * @param nms_threshold NMS阈值
 * @param max_objects 最大对象数量
 * @param stream CUDA流
 */
void cuda_nms(float* parray, float nms_threshold, int max_objects,
              cudaStream_t stream) {
  // 设置CUDA执行配置
  // 块大小不超过256，也不超过最大对象数量
  int block = max_objects < 256 ? max_objects : 256;
  int grid = ceil(max_objects / (float)block);  // 块的数量

  // 启动NMS核函数
  nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold);
}

}  // extern "C"