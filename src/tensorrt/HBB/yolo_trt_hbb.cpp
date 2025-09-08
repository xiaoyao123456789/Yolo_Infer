#include "yolo_trt_hbb.hpp"

#include <cassert>  // 用于assert
#include <cstdio>
#include <filesystem>
#include <fstream>  // 用于std::ifstream
#include <iostream>
#include <opencv2/opencv.hpp>

#include "cuda_utils.hpp"
#include "postprocess.h"
#include "preprocess.h"
#include "trt_utils.hpp"

using namespace nvinfer1;

// 主要的TensorRT水平边界框(HBB)检测函数
void maintrt_run_hbb(const std::string &imagePath,
                     const std::wstring &modelPath,
                     const std::string &outputPath) {
  // 将wstring转换为string
  std::string engine_name(modelPath.begin(), modelPath.end());

  // TensorRT相关变量初始化
  IRuntime *runtime = nullptr;
  ICudaEngine *engine = nullptr;
  IExecutionContext *context = nullptr;

  // 反序列化引擎文件
  deserialize_engine(engine_name, &runtime, &engine, &context);

  std::cout << "Engine name: " << engine_name << std::endl;
  std::cout << "Number of bindings: " << engine->getNbBindings() << std::endl;
  cv::Mat image = read_images(imagePath);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cuda_preprocess_init(max_image_size);
  auto out_dims = engine->getBindingDimensions(1);
  auto model_bboxes = out_dims.d[2];
  for (int i = 0; i < out_dims.nbDims; i++) {
    printf("%d ", out_dims.d[i]);
  }
  printf("\n");
  printf("model_bboxes: %d\n", model_bboxes);
  float *device_buffers[2];
  float *output_buffer_host = nullptr;
  float *decode_ptr_host = nullptr;
  float *decode_ptr_device = nullptr;

  prepare_buffer(engine, &device_buffers[0], &device_buffers[1],
                 &output_buffer_host, &decode_ptr_host, &decode_ptr_device,
                 "g");

  cuda_preprocess(image.data, image.cols, image.rows, device_buffers[0],
                  kInputW, kInputH, stream);
  // 保存预处理结果用于调试
  save_warpaffine_image(device_buffers[0], kInputW, kInputH,
                        "warpaffine_result.jpg", stream);

  // 分配临时缓冲区保存原始输出
  auto output_dims = engine->getBindingDimensions(1);
  int output_size = 1;
  for (int i = 0; i < output_dims.nbDims; i++) {
    output_size *= output_dims.d[i];
  }
  float *temp_output_host = new float[output_size];

  infer(*context, stream, (void **)device_buffers, temp_output_host, 1,
        decode_ptr_host, decode_ptr_device, model_bboxes, "g");

  // 释放临时缓冲区
  delete[] temp_output_host;

  int num_detections = static_cast<int>(decode_ptr_host[0]);
  std::cout << "Number of detections: " << num_detections << std::endl;

  // 在原图上绘制检测结果
  cv::Mat result_image = image.clone();

  std::cout << "Final detections after CPU post-processing: " << num_detections
            << std::endl;

  // CPU NMS后，parray[0]是保留的检测框数量，但检测框位置没有改变
  // 需要遍历所有检测框并检查keep标志
  int total_boxes = 0;
  // 先找到总的检测框数量（在NMS之前）
  for (int i = 0; i < 1000; i++) {  // 假设最大1000个框
    int base_idx = 1 + i * 7;
    if (base_idx + 6 >= (1 + 1000 * 7)) break;  // 防止越界
    float conf = decode_ptr_host[base_idx + 4];
    if (conf <= 0) break;  // 遇到无效框就停止
    total_boxes++;
  }

  std::cout << "Total boxes to check: " << total_boxes << std::endl;

  int drawn_count = 0;
  for (int i = 0; i < total_boxes; i++) {
    int base_idx =
        1 +
        i * 7;  // CPU后处理输出7个元素: [x1, y1, x2, y2, conf, cls, keep_flag]

    // CPU后处理输出格式：[x1, y1, x2, y2, conf, cls, keep_flag]
    float x1 = decode_ptr_host[base_idx + 0];  // 左上角x坐标（已经是原图坐标）
    float y1 = decode_ptr_host[base_idx + 1];  // 左上角y坐标（已经是原图坐标）
    float x2 = decode_ptr_host[base_idx + 2];  // 右下角x坐标（已经是原图坐标）
    float y2 = decode_ptr_host[base_idx + 3];  // 右下角y坐标（已经是原图坐标）
    float conf = decode_ptr_host[base_idx + 4];                  // 置信度
    int cls = static_cast<int>(decode_ptr_host[base_idx + 5]);   // 类别
    int keep = static_cast<int>(decode_ptr_host[base_idx + 6]);  // keep标志

    std::cout << "Box " << i << ": [" << x1 << ", " << y1 << ", " << x2 << ", "
              << y2 << "], conf=" << conf << ", cls=" << cls
              << ", keep=" << keep << std::endl;

    // 只绘制keep=1的检测框
    if (keep == 1) {
      std::cout << "Drawing detection " << drawn_count << ": [" << x1 << ", "
                << y1 << ", " << x2 << ", " << y2 << "], conf=" << conf
                << ", cls=" << cls << std::endl;

      // 绘制边界框
      cv::Rect bbox(static_cast<int>(x1), static_cast<int>(y1),
                    static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
      cv::rectangle(result_image, bbox, cv::Scalar(0, 255, 0), 2);

      // 绘制标签和置信度
      std::string label = "Det" + std::to_string(cls) + ": " +
                          std::to_string(conf).substr(0, 4);
      cv::putText(result_image, label,
                  cv::Point(static_cast<int>(x1), static_cast<int>(y1) - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

      drawn_count++;
    }
  }

  std::cout << "Total boxes drawn: " << drawn_count << std::endl;

  // 保存结果图像
  if (!outputPath.empty()) {
    cv::imwrite(outputPath, result_image);
    std::cout << "Result saved to: " << outputPath << std::endl;
  } else {
    // 如果没有指定输出路径，保存到默认位置
    std::string default_output = "result.jpg";
    cv::imwrite(default_output, result_image);
    std::cout << "Result saved to: " << default_output << std::endl;
  }

  // 清理资源 - 先确保所有CUDA操作完成
  std::cout << "[DEBUG] Starting cleanup..." << std::endl;
  if (stream) {
    std::cout << "[DEBUG] Synchronizing CUDA stream..." << std::endl;
    CUDA_CHECK(cudaStreamSynchronize(stream));  // 确保所有操作完成
    std::cout << "[DEBUG] Destroying CUDA stream..." << std::endl;
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "[DEBUG] CUDA stream destroyed successfully" << std::endl;
  }
  std::cout << "[DEBUG] Calling cuda_preprocess_destroy..." << std::endl;
  cuda_preprocess_destroy();
  std::cout << "[DEBUG] cuda_preprocess_destroy completed" << std::endl;

  // 释放设备内存 - 先释放设备端内存
  if (decode_ptr_device) {
    std::cout << "[DEBUG] Freeing decode_ptr_device..." << std::endl;
    CUDA_CHECK(cudaFree(decode_ptr_device));
    std::cout << "[DEBUG] decode_ptr_device freed successfully" << std::endl;
  }

  // 添加指针有效性检查
  std::cout << "[DEBUG] device_buffers[0] pointer: " << device_buffers[0]
            << std::endl;
  std::cout << "[DEBUG] device_buffers[1] pointer: " << device_buffers[1]
            << std::endl;

  if (device_buffers[0] && device_buffers[0] != (float *)0xDDDDDDDD &&
      device_buffers[0] != (float *)0xCCCCCCCC) {
    std::cout << "[DEBUG] Freeing device_buffers[0]..." << std::endl;
    CUDA_CHECK(cudaFree(device_buffers[0]));
    std::cout << "[DEBUG] device_buffers[0] freed successfully" << std::endl;
    device_buffers[0] = nullptr;
  } else {
    std::cout << "[DEBUG] device_buffers[0] is invalid, skipping..."
              << std::endl;
  }

  if (device_buffers[1] && device_buffers[1] != (float *)0xDDDDDDDD &&
      device_buffers[1] != (float *)0xCCCCCCCC) {
    std::cout << "[DEBUG] Freeing device_buffers[1]..." << std::endl;
    CUDA_CHECK(cudaFree(device_buffers[1]));
    std::cout << "[DEBUG] device_buffers[1] freed successfully" << std::endl;
    device_buffers[1] = nullptr;
  } else {
    std::cout << "[DEBUG] device_buffers[1] is invalid, skipping..."
              << std::endl;
  }

  // 释放主机内存 - 后释放主机端内存
  std::cout << "[DEBUG] output_buffer_host pointer: " << output_buffer_host
            << std::endl;
  std::cout << "[DEBUG] decode_ptr_host pointer: " << decode_ptr_host
            << std::endl;

  if (output_buffer_host && output_buffer_host != (float *)0xDDDDDDDD &&
      output_buffer_host != (float *)0xCCCCCCCC) {
    std::cout << "[DEBUG] Deleting output_buffer_host..." << std::endl;
    delete[] output_buffer_host;
    std::cout << "[DEBUG] output_buffer_host deleted successfully" << std::endl;
    output_buffer_host = nullptr;
  } else {
    std::cout << "[DEBUG] output_buffer_host is invalid, skipping..."
              << std::endl;
  }

  if (decode_ptr_host && decode_ptr_host != (float *)0xDDDDDDDD &&
      decode_ptr_host != (float *)0xCCCCCCCC) {
    std::cout << "[DEBUG] Deleting decode_ptr_host..." << std::endl;
    delete[] decode_ptr_host;  // 使用delete[]释放new分配的内存
    std::cout << "[DEBUG] decode_ptr_host deleted successfully" << std::endl;
    decode_ptr_host = nullptr;
  } else {
    std::cout << "[DEBUG] decode_ptr_host is invalid, skipping..." << std::endl;
  }

  // 销毁TensorRT对象 - 按正确顺序销毁
  if (context) {
    std::cout << "[DEBUG] Destroying TensorRT context..." << std::endl;
    context->destroy();
    std::cout << "[DEBUG] TensorRT context destroyed successfully" << std::endl;
  }
  if (engine) {
    std::cout << "[DEBUG] Destroying TensorRT engine..." << std::endl;
    engine->destroy();
    std::cout << "[DEBUG] TensorRT engine destroyed successfully" << std::endl;
  }
  if (runtime) {
    std::cout << "[DEBUG] Destroying TensorRT runtime..." << std::endl;
    runtime->destroy();
    std::cout << "[DEBUG] TensorRT runtime destroyed successfully" << std::endl;
  }

  // 确保CUDA上下文同步
  std::cout << "[DEBUG] Final CUDA device synchronization..." << std::endl;
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "[DEBUG] All cleanup completed successfully" << std::endl;
}