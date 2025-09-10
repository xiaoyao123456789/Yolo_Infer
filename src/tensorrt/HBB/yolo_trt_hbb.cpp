#include "yolo_trt_hbb.hpp"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "cuda_utils.hpp"
#include "postprocess.h"
#include "preprocess.h"
#include "trt_utils.hpp"

using namespace nvinfer1;

// 主要的TensorRT水平边界框(HBB)检测函数
// 支持单张图片或批量处理（传入目录路径）
// 参数说明：
// - inputPath: 输入图片路径或目录路径
// - modelPath: TensorRT引擎模型文件路径（wstring类型，支持中文路径）
// - outputPath: 输出结果保存路径
void maintrt_run_hbb(const std::string &inputPath,
                     const std::wstring &modelPath,
                     const std::string &outputPath) {
  // 检查输入路径是文件还是目录
  bool isDirectory = std::filesystem::is_directory(inputPath);
  std::vector<std::string> imageFiles;

  if (isDirectory) {
    // 批量处理：获取目录中的所有图片文件
    std::vector<std::string> supportedExtensions = {".jpg", ".jpeg", ".png",
                                                    ".bmp", ".tiff"};

    // 遍历目录中的所有文件
    for (const auto &entry : std::filesystem::directory_iterator(inputPath)) {
      if (entry.is_regular_file()) {
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       ::tolower);

        // 检查文件扩展名是否在支持的图片格式列表中
        if (std::find(supportedExtensions.begin(), supportedExtensions.end(),
                      extension) != supportedExtensions.end()) {
          imageFiles.push_back(entry.path().string());
        }
      }
    }

    // 如果没有找到图片文件，直接返回
    if (imageFiles.empty()) {
      std::cout << "No image files found in directory: " << inputPath
                << std::endl;
      return;
    }

    std::cout << "Found " << imageFiles.size() << " image files to process"
              << std::endl;

    // 创建输出目录（如果不存在）
    if (!std::filesystem::exists(outputPath)) {
      std::filesystem::create_directories(outputPath);
      std::cout << "Created output directory: " << outputPath << std::endl;
    }
  } else {
    // 单张图片处理
    if (!std::filesystem::exists(inputPath)) {
      std::cerr << "Error: Input file does not exist: " << inputPath
                << std::endl;
      return;
    }
    imageFiles.push_back(inputPath);
  }

  // 将wstring转换为string（处理中文路径问题）
  std::string engine_name(modelPath.begin(), modelPath.end());

  // TensorRT相关变量初始化（只初始化一次，避免重复创建）
  IRuntime *runtime = nullptr;           // TensorRT运行时环境
  ICudaEngine *engine = nullptr;         // TensorRT引擎
  IExecutionContext *context = nullptr;  // 执行上下文

  // 反序列化引擎文件（从文件加载预编译的TensorRT引擎）
  deserialize_engine(engine_name, &runtime, &engine, &context);

  // 初始化CUDA资源（只初始化一次）
  cudaStream_t stream;  // CUDA流，用于异步执行
  CUDA_CHECK(cudaStreamCreate(&stream));
  cuda_preprocess_init(max_image_size);  // 预处理初始化

  // 获取输出张量的维度信息
  auto out_dims = engine->getBindingDimensions(1);
  auto model_bboxes = out_dims.d[2];  // 模型支持的最大检测框数量

  // GPU缓冲区指针
  float *device_buffers[2];             // 设备端缓冲区数组
  float *output_buffer_host = nullptr;  // 主机端输出缓冲区
  float *decode_ptr_host = nullptr;     // 主机端解码缓冲区
  float *decode_ptr_device = nullptr;   // 设备端解码缓冲区

  // 准备缓冲区（分配GPU和CPU内存）
  prepare_buffer(engine, &device_buffers[0], &device_buffers[1],
                 &output_buffer_host, &decode_ptr_host, &decode_ptr_device,
                 "g");

  // 分配临时缓冲区保存原始输出
  auto output_dims = engine->getBindingDimensions(1);
  int output_size = 1;
  // 计算输出张量的总元素数量
  for (int i = 0; i < output_dims.nbDims; i++) {
    output_size *= output_dims.d[i];
  }
  float *temp_output_host = new float[output_size];

  // Warmup阶段：使用第一张图片进行3次warmup推理
  // 目的是让GPU达到稳定状态，避免首次推理的延迟影响性能测试
  if (!imageFiles.empty()) {
    std::cout << "Starting warmup with 3 iterations..." << std::endl;
    cv::Mat warmup_image = read_images(imageFiles[0]);
    if (!warmup_image.empty()) {
      for (int i = 0; i < 3; ++i) {
        // 重置状态，确保每次推理都是独立的
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 清空检测框缓冲区
        memset(decode_ptr_host, 0, (1 + 1000 * 7) * sizeof(float));
        CUDA_CHECK(
            cudaMemset(decode_ptr_device, 0, (1 + 1000 * 7) * sizeof(float)));

        // Warmup推理过程
        cuda_preprocess(warmup_image.data, warmup_image.cols, warmup_image.rows,
                        device_buffers[0], kInputW, kInputH, stream);
        infer(*context, stream, (void **)device_buffers, temp_output_host, 1,
              decode_ptr_host, decode_ptr_device, model_bboxes, "g",
              warmup_image.cols, warmup_image.rows);

        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      warmup_image.release();  // 释放warmup图片内存
    }
    std::cout << "Warmup completed!" << std::endl;
  }

  // 处理每张图片的主循环
  int processed_count = 0;  // 已处理图片计数
  int success_count = 0;    // 成功处理图片计数

  for (const auto &imagePath : imageFiles) {
    try {
      // 批量处理时显示进度信息
      if (isDirectory) {
        std::cout << "Processing [" << (processed_count + 1) << "/"
                  << imageFiles.size() << "] "
                  << std::filesystem::path(imagePath).filename().string()
                  << std::endl;
      }

      // 读取图像
      cv::Mat image = read_images(imagePath);
      if (image.empty()) {
        std::cerr << "Error: Could not read image: " << imagePath << std::endl;
        processed_count++;
        continue;
      }

      // 重置所有状态，确保每次推理都是独立的
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // 清空检测框缓冲区 - 确保完全清空之前的检测数据
      memset(decode_ptr_host, 0, (1 + 1000 * 7) * sizeof(float));
      CUDA_CHECK(
          cudaMemset(decode_ptr_device, 0, (1 + 1000 * 7) * sizeof(float)));

      // 强制同步确保清空完成
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // 开始计时，测量推理耗时
      auto start_time = std::chrono::high_resolution_clock::now();

      // 预处理阶段：将图片resize到模型输入尺寸，并进行归一化等操作
      cuda_preprocess(image.data, image.cols, image.rows, device_buffers[0],
                      kInputW, kInputH, stream);

      // 执行推理：将预处理后的数据送入TensorRT引擎进行前向计算
      infer(*context, stream, (void **)device_buffers, temp_output_host, 1,
            decode_ptr_host, decode_ptr_device, model_bboxes, "g", image.cols,
            image.rows);

      // 结束计时
      auto end_time = std::chrono::high_resolution_clock::now();
      auto inference_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                start_time)
              .count();

      // 获取检测框数量（decode_ptr_host[0]存储了检测到的目标数量）
      int num_detections = static_cast<int>(decode_ptr_host[0]);

      // 在原图上绘制检测结果
      cv::Mat result_image = image.clone();

      // 遍历所有检测框并绘制
      int drawn_count = 0;  // 实际绘制的检测框数量
      for (int i = 0; i < num_detections; i++) {
        // 计算当前检测框在缓冲区中的起始索引
        int base_idx = 1 + i * 7;

        // 提取检测框信息
        float x1 = decode_ptr_host[base_idx + 0];    // 左上角x坐标
        float y1 = decode_ptr_host[base_idx + 1];    // 左上角y坐标
        float x2 = decode_ptr_host[base_idx + 2];    // 右下角x坐标
        float y2 = decode_ptr_host[base_idx + 3];    // 右下角y坐标
        float conf = decode_ptr_host[base_idx + 4];  // 置信度
        int cls = static_cast<int>(decode_ptr_host[base_idx + 5]);  // 类别ID
        int keep =
            static_cast<int>(decode_ptr_host[base_idx + 6]);  // 是否保留标志

        // 只绘制keep=1的检测框（通过NMS过滤后的有效框）
        if (keep == 1) {
          // 绘制矩形框
          cv::Rect bbox(static_cast<int>(x1), static_cast<int>(y1),
                        static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
          cv::rectangle(result_image, bbox, cv::Scalar(0, 255, 0), 2);

          // 绘制类别标签和置信度
          std::string label = "Det" + std::to_string(cls) + ": " +
                              std::to_string(conf).substr(0, 4);
          cv::putText(
              result_image, label,
              cv::Point(static_cast<int>(x1), static_cast<int>(y1) - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
          drawn_count++;
        }
      }

      // 生成输出文件路径
      std::string finalOutputPath;
      if (isDirectory) {
        // 批量处理：根据输入文件名生成输出文件名
        std::filesystem::path inputPath(imagePath);
        std::string outputFileName = inputPath.stem().string() + "_result" +
                                     inputPath.extension().string();
        finalOutputPath =
            (std::filesystem::path(outputPath) / outputFileName).string();
      } else {
        // 单张图片处理：处理输出路径逻辑
        if (outputPath.empty()) {
          finalOutputPath = "result.jpg";  // 默认输出文件名
        } else if (std::filesystem::is_directory(outputPath) ||
                   outputPath.back() == '/' || outputPath.back() == '\\') {
          // outputPath是目录，生成文件名
          std::filesystem::path inputPath(imagePath);
          std::string outputFileName = inputPath.stem().string() + "_result" +
                                       inputPath.extension().string();
          finalOutputPath =
              (std::filesystem::path(outputPath) / outputFileName).string();
        } else {
          // outputPath是完整文件路径，直接使用
          finalOutputPath = outputPath;
        }
      }

      // 保存结果图像
      if (cv::imwrite(finalOutputPath, result_image)) {
        std::cout << "Inference time: " << inference_time << "ms, "
                  << "Detections: " << drawn_count << " - " << finalOutputPath
                  << std::endl;
        success_count++;
      } else {
        std::cerr << "Error: Failed to save result to: " << finalOutputPath
                  << std::endl;
      }

      // 确保所有GPU操作完成
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaDeviceSynchronize());

      // 重置缓冲区状态，为下一张图片做准备
      memset(decode_ptr_host, 0, (1 + 1000 * 7) * sizeof(float));
      CUDA_CHECK(
          cudaMemset(decode_ptr_device, 0, (1 + 1000 * 7) * sizeof(float)));

    } catch (const std::exception &e) {
      // 异常处理：捕获并报告处理过程中的错误
      std::cerr << "Error processing " << imagePath << ": " << e.what()
                << std::endl;
    }

    processed_count++;  // 增加已处理计数
  }

  // 释放临时缓冲区内存
  delete[] temp_output_host;

  // 批量处理完成后的总结信息
  if (isDirectory) {
    std::cout << "Batch processing completed: " << success_count << "/"
              << processed_count << " successful" << std::endl;
  }

  // ===== 清理资源阶段 =====
  // 释放CUDA流
  if (stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));  // 确保所有操作完成
    CUDA_CHECK(cudaStreamDestroy(stream));      // 销毁CUDA流
  }
  cuda_preprocess_destroy();  // 清理预处理相关资源

  // 释放GPU内存
  if (decode_ptr_device) {
    CUDA_CHECK(cudaFree(decode_ptr_device));
  }

  if (device_buffers[0]) {
    CUDA_CHECK(cudaFree(device_buffers[0]));
  }

  if (device_buffers[1]) {
    CUDA_CHECK(cudaFree(device_buffers[1]));
  }

  // 释放CPU内存
  if (output_buffer_host) {
    delete[] output_buffer_host;
  }

  if (decode_ptr_host) {
    delete[] decode_ptr_host;
  }

  // 释放TensorRT资源
  if (context) context->destroy();
  if (engine) engine->destroy();
  if (runtime) runtime->destroy();

  // 最终同步，确保所有GPU操作完成
  CUDA_CHECK(cudaDeviceSynchronize());
}