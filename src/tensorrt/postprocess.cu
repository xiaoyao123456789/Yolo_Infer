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
 * @brief 坐标变换CUDA核函数
 * @param parray 检测结果数组
 * @param img_width 原始图像宽度
 * @param img_height 原始图像高度
 * @param input_width 模型输入宽度
 * @param input_height 模型输入高度
 * @param max_objects 最大对象数量
 */
__global__ void coordinate_transform_kernel(float* parray, int img_width,
                                            int img_height, int input_width,
                                            int input_height, int max_objects);

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
/**
 * @brief 快速IoU计算函数（优化版本）
 * 
 * 直接使用边界框坐标计算IoU，避免坐标转换开销
 * 
 * @param box1 第一个边界框 [x1, y1, x2, y2, conf, class, keep]
 * @param box2 第二个边界框 [x1, y1, x2, y2, conf, class, keep]
 * @return IoU值
 */
__device__ __forceinline__ float fast_box_iou(const float* box1, const float* box2) {
  // 直接使用边界框坐标，避免中心坐标转换
  float left = fmaxf(box1[0], box2[0]);
  float top = fmaxf(box1[1], box2[1]);
  float right = fminf(box1[2], box2[2]);
  float bottom = fminf(box1[3], box2[3]);
  
  // 快速检查是否有交集
  if (left >= right || top >= bottom) {
    return 0.0f;
  }
  
  // 计算交集面积
  float intersection = (right - left) * (bottom - top);
  
  // 计算两个框的面积
  float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
  float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
  
  // 计算并集面积
  float union_area = area1 + area2 - intersection;
  
  // 返回IoU值，使用更小的epsilon值
  return intersection / (union_area + 1e-8f);
}

/**
 * @brief 原始IoU计算函数（保持向后兼容）
 */
__device__ float box_probiou(float ax, float ay, float aw, float ah,
                             float aangle, float bx, float by, float bw,
                             float bh, float bangle) {
  float left = fmaxf(ax - aw / 2, bx - bw / 2);
  float right = fminf(ax + aw / 2, bx + bw / 2);
  float top = fmaxf(ay - ah / 2, by - bh / 2);
  float bottom = fminf(ay + ah / 2, by + bh / 2);
  
  float width = fmaxf(0.0f, right - left);
  float height = fmaxf(0.0f, bottom - top);
  
  float intersection = width * height;
  float area_a = aw * ah;
  float area_b = bw * bh;
  float union_area = area_a + area_b - intersection;
  
  return intersection / (union_area + 1e-7f);
}

// ============================================================================
// CUDA核函数实现
// ============================================================================

// 声明外部定义的常量
extern const int bbox_element;

/**
 * @brief 优化的非极大值抑制（NMS）核函数
 *
 * 使用共享内存和优化的内存访问模式来提高NMS性能。
 * 每个线程块处理一批检测框，使用共享内存减少全局内存访问。
 *
 * @param parray 检测结果数组，格式：[count, bbox1, bbox2, ...]
 *               每个bbox格式：[x1, y1, x2, y2, confidence, class_id, keep_flag]
 * @param max_objects 最大对象数量
 * @param threshold NMS阈值，通常为0.5
 */
__global__ void nms_kernel_optimized(float* parray, int max_objects, float threshold) {
  // 共享内存用于缓存当前块处理的检测框
  extern __shared__ float shared_boxes[];
  
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int block_size = blockDim.x;
  int position = bid * block_size + tid;
  
  // 获取实际检测框数量
  int count = min((int)parray[0], max_objects);
  
  if (position >= count) return;
  
  // 将当前块需要处理的检测框加载到共享内存
  int boxes_per_block = min(block_size, count - bid * block_size);
  
  // 每个线程加载一个检测框到共享内存
  if (tid < boxes_per_block) {
    int src_idx = bid * block_size + tid;
    if (src_idx < count) {
      float* src_box = parray + 1 + src_idx * bbox_element;
      float* dst_box = shared_boxes + tid * bbox_element;
      
      // 使用向量化加载提高内存带宽利用率
      #pragma unroll
      for (int i = 0; i < bbox_element; i++) {
        dst_box[i] = src_box[i];
      }
    }
  }
  
  __syncthreads();
  
  // 获取当前线程处理的检测框
  float* pcurrent = parray + 1 + position * bbox_element;
  
  // 首先与共享内存中的框进行比较（块内比较）
  for (int i = 0; i < boxes_per_block; i++) {
    if (i == tid) continue;  // 跳过自己
    
    float* pitem = shared_boxes + i * bbox_element;
    
    // 跳过不同类别的框
    if (pcurrent[5] != pitem[5]) continue;
    
    // 置信度比较
    if (pitem[4] >= pcurrent[4]) {
      if (pitem[4] == pcurrent[4] && (bid * block_size + i) < position) continue;
      
      // 快速IoU计算
      float iou = fast_box_iou(pcurrent, pitem);
      
      if (iou > threshold) {
        pcurrent[6] = 0;
        return;
      }
    }
  }
  
  // 然后与其他块的框进行比较（全局比较）
  for (int block_id = 0; block_id < gridDim.x; block_id++) {
    if (block_id == bid) continue;  // 跳过当前块
    
    int start_idx = block_id * block_size;
    int end_idx = min(start_idx + block_size, count);
    
    for (int i = start_idx; i < end_idx; i++) {
      float* pitem = parray + 1 + i * bbox_element;
      
      // 跳过不同类别的框
      if (pcurrent[5] != pitem[5]) continue;
      
      // 置信度比较
      if (pitem[4] >= pcurrent[4]) {
        if (pitem[4] == pcurrent[4] && i < position) continue;
        
        // 快速IoU计算
        float iou = fast_box_iou(pcurrent, pitem);
        
        if (iou > threshold) {
          pcurrent[6] = 0;
          return;
        }
      }
    }
  }
}

/**
 * @brief 原始NMS核函数（保持向后兼容）
 */
__global__ void nms_kernel(float* parray, int max_objects, float threshold) {
  int position = (blockDim.x * blockIdx.x + threadIdx.x);
  int count = min((int)parray[0], max_objects);
  
  if (position >= count) return;
  
  float* pcurrent = parray + 1 + position * bbox_element;
  
  for (int i = 0; i < count; ++i) {
    float* pitem = parray + 1 + i * bbox_element;
    
    if (i == position || pcurrent[5] != pitem[5]) continue;
    
    if (pitem[4] >= pcurrent[4]) {
      if (pitem[4] == pcurrent[4] && i < position) continue;
      
      float iou = fast_box_iou(pcurrent, pitem);
      
      if (iou > threshold) {
        pcurrent[6] = 0;
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

      float cx2 = (pitem[0] + pitem[2]) * 0.5f;  // 比较框中心x
      float cy2 = (pitem[1] + pitem[3]) * 0.5f;  // 比较框中心y
      float w2 = pitem[2] - pitem[0];            // 比较框宽度
      float h2 = pitem[3] - pitem[1];            // 比较框高度

      // pcurrent[7]和pitem[7]分别是当前框和比较框的旋转角度
      float iou = box_probiou(cx1, cy1, w1, h1, pcurrent[7], cx2, cy2, w2, h2,
                              pitem[7]);

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

  // 按照CPU版本的数据布局访问：predict[channel * num_bboxes + position]
  // 添加边界检查防止数组越界
  int base_idx = position;
  if (base_idx >= num_bboxes) {
    return;
  }

  float cx = predict[0 * num_bboxes + base_idx];          // 中心x坐标
  float cy = predict[1 * num_bboxes + base_idx];          // 中心y坐标
  float w = predict[2 * num_bboxes + base_idx];           // 宽度
  float h = predict[3 * num_bboxes + base_idx];           // 高度
  float confidence = predict[4 * num_bboxes + base_idx];  // 置信度

  int label = 0;  // 默认类别标签
  float max_conf = confidence;

  // 多类别情况：寻找置信度最高的类别
  if (num_class > 1) {
    for (int c = 1; c < num_class; ++c) {
      float conf =
          predict[(4 + c) * num_bboxes + position];  // 获取第c类的置信度
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

/**
 * @brief 坐标变换CUDA核函数实现
 *
 * 将模型坐标系的检测框坐标转换回原图坐标系。
 * 每个线程处理一个检测框的坐标变换。
 *
 * @param parray 检测结果数组，格式：[count, bbox1, bbox2, ...]
 * @param img_width 原始图像宽度
 * @param img_height 原始图像高度
 * @param input_width 模型输入宽度
 * @param input_height 模型输入高度
 * @param max_objects 最大对象数量
 */
__global__ void coordinate_transform_kernel(float* parray, int img_width,
                                            int img_height, int input_width,
                                            int input_height, int max_objects) {
  // 计算当前线程处理的检测框索引
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 获取实际检测框数量
  int count = (int)parray[0];

  // 边界检查
  if (idx >= count || idx >= max_objects) return;

  // 计算缩放参数（与CPU版本保持一致）
  float scale = fminf((float)input_height / (float)img_height,
                      (float)input_width / (float)img_width);
  float dx = ((float)input_width - scale * (float)img_width) / 2.0f;
  float dy = ((float)input_height - scale * (float)img_height) / 2.0f;

  // 计算当前检测框在数组中的位置（每个框7个元素：x1,y1,x2,y2,conf,class,keep）
  float* box = parray + 1 + idx * 7;

  // 读取当前坐标
  float x1 = box[0];
  float y1 = box[1];
  float x2 = box[2];
  float y2 = box[3];

  // 执行逆变换：从模型坐标系转换回原图坐标系
  float new_x1 = (x1 - dx) / scale;
  float new_y1 = (y1 - dy) / scale;
  float new_x2 = (x2 - dx) / scale;
  float new_y2 = (y2 - dy) / scale;

  // 写回变换后的坐标
  box[0] = new_x1;
  box[1] = new_y1;
  box[2] = new_x2;
  box[3] = new_y2;
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
 * @brief 优化的CUDA NMS函数的C接口
 *
 * 使用共享内存优化的NMS核函数，提供更好的性能。
 *
 * @param parray 检测结果数组指针
 * @param nms_threshold NMS阈值
 * @param max_objects 最大对象数量
 * @param stream CUDA流
 * @param use_optimized 是否使用优化版本（默认true）
 */
void cuda_nms_optimized(float* parray, float nms_threshold, int max_objects,
                       cudaStream_t stream, bool use_optimized) {
  if (use_optimized) {
    // 优化版本：使用共享内存
    int block = 128;  // 优化的块大小
    int grid = ceil(max_objects / (float)block);
    
    // 计算共享内存大小：每个块最多处理128个框，每个框7个元素
    int shared_mem_size = block * 7 * sizeof(float);
    
    // 启动优化的NMS核函数
    nms_kernel_optimized<<<grid, block, shared_mem_size, stream>>>(
        parray, max_objects, nms_threshold);
  } else {
    // 原始版本
    cuda_nms(parray, nms_threshold, max_objects, stream);
  }
}

/**
 * @brief 分层NMS函数
 *
 * 先进行粗筛选，再进行精确NMS，提高大量检测框场景下的性能。
 *
 * @param parray 检测结果数组指针
 * @param nms_threshold NMS阈值
 * @param max_objects 最大对象数量
 * @param stream CUDA流
 * @param coarse_threshold 粗筛选阈值（通常比nms_threshold大0.1-0.2）
 */
void cuda_nms_hierarchical(float* parray, float nms_threshold, int max_objects,
                          cudaStream_t stream, float coarse_threshold) {
  // 第一阶段：粗筛选，使用较高的阈值快速去除明显重叠的框
  cuda_nms_optimized(parray, coarse_threshold, max_objects, stream, true);
  
  // 同步确保第一阶段完成
  cudaStreamSynchronize(stream);
  
  // 第二阶段：精确NMS，使用目标阈值进行精确筛选
  cuda_nms_optimized(parray, nms_threshold, max_objects, stream, true);
}

/**
 * @brief 原始CUDA NMS函数的C接口（保持向后兼容）
 */
void cuda_nms(float* parray, float nms_threshold, int max_objects,
              cudaStream_t stream) {
  int block = max_objects < 256 ? max_objects : 256;
  int grid = ceil(max_objects / (float)block);
  
  nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold);
}

/**
 * @brief CUDA坐标变换函数的C接口
 *
 * 启动CUDA核函数进行坐标变换，将模型坐标系转换回原图坐标系。
 *
 * @param parray 检测结果数组指针
 * @param img_width 原始图像宽度
 * @param img_height 原始图像高度
 * @param input_width 模型输入宽度
 * @param input_height 模型输入高度
 * @param max_objects 最大对象数量
 * @param stream CUDA流
 */
void cuda_coordinate_transform(float* parray, int img_width, int img_height,
                               int input_width, int input_height,
                               int max_objects, cudaStream_t stream) {
  // 设置CUDA执行配置
  int block = 256;                              // 每个块的线程数
  int grid = ceil(max_objects / (float)block);  // 块的数量

  // 启动坐标变换核函数
  coordinate_transform_kernel<<<grid, block, 0, stream>>>(
      parray, img_width, img_height, input_width, input_height, max_objects);
}

}  // extern "C"