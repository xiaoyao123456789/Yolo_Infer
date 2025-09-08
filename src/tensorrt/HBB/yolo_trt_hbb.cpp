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

    for (const auto &entry : std::filesystem::directory_iterator(inputPath)) {
      if (entry.is_regular_file()) {
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       ::tolower);

        if (std::find(supportedExtensions.begin(), supportedExtensions.end(),
                      extension) != supportedExtensions.end()) {
          imageFiles.push_back(entry.path().string());
        }
      }
    }

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

  // 将wstring转换为string
  std::string engine_name(modelPath.begin(), modelPath.end());

  // TensorRT相关变量初始化（只初始化一次）
  IRuntime *runtime = nullptr;
  ICudaEngine *engine = nullptr;
  IExecutionContext *context = nullptr;

  // 反序列化引擎文件
  deserialize_engine(engine_name, &runtime, &engine, &context);

  // 初始化CUDA资源（只初始化一次）
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cuda_preprocess_init(max_image_size);
  auto out_dims = engine->getBindingDimensions(1);
  auto model_bboxes = out_dims.d[2];
  float *device_buffers[2];
  float *output_buffer_host = nullptr;
  float *decode_ptr_host = nullptr;
  float *decode_ptr_device = nullptr;

  // 准备缓冲区（只准备一次）
  prepare_buffer(engine, &device_buffers[0], &device_buffers[1],
                 &output_buffer_host, &decode_ptr_host, &decode_ptr_device,
                 "g");

  // 分配临时缓冲区保存原始输出
  auto output_dims = engine->getBindingDimensions(1);
  int output_size = 1;
  for (int i = 0; i < output_dims.nbDims; i++) {
    output_size *= output_dims.d[i];
  }
  float *temp_output_host = new float[output_size];

  // 处理每张图片
  int processed_count = 0;
  int success_count = 0;

  for (const auto &imagePath : imageFiles) {
    try {
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

      // Reset all states before each inference
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // Clear buffers
      memset(decode_ptr_host, 0, (1 + 1000 * 7) * sizeof(float));
      CUDA_CHECK(
          cudaMemset(decode_ptr_device, 0, (1 + 1000 * 7) * sizeof(float)));

      // Start timing
      auto start_time = std::chrono::high_resolution_clock::now();

      // Preprocessing
      cuda_preprocess(image.data, image.cols, image.rows, device_buffers[0],
                      kInputW, kInputH, stream);

      // Execute inference
      infer(*context, stream, (void **)device_buffers, temp_output_host, 1,
            decode_ptr_host, decode_ptr_device, model_bboxes, "g", image.cols,
            image.rows);

      // End timing
      auto end_time = std::chrono::high_resolution_clock::now();
      auto inference_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                start_time)
              .count();

      int num_detections = static_cast<int>(decode_ptr_host[0]);

      // 在原图上绘制检测结果
      cv::Mat result_image = image.clone();

      int drawn_count = 0;
      for (int i = 0; i < num_detections; i++) {
        int base_idx = 1 + i * 7;

        float x1 = decode_ptr_host[base_idx + 0];
        float y1 = decode_ptr_host[base_idx + 1];
        float x2 = decode_ptr_host[base_idx + 2];
        float y2 = decode_ptr_host[base_idx + 3];
        float conf = decode_ptr_host[base_idx + 4];
        int cls = static_cast<int>(decode_ptr_host[base_idx + 5]);
        int keep = static_cast<int>(decode_ptr_host[base_idx + 6]);

        // 只绘制keep=1的检测框
        if (keep == 1) {
          cv::Rect bbox(static_cast<int>(x1), static_cast<int>(y1),
                        static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
          cv::rectangle(result_image, bbox, cv::Scalar(0, 255, 0), 2);

          std::string label = "Det" + std::to_string(cls) + ": " +
                              std::to_string(conf).substr(0, 4);
          cv::putText(
              result_image, label,
              cv::Point(static_cast<int>(x1), static_cast<int>(y1) - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
          drawn_count++;
        }
      }

      // 保存结果图像
      std::string finalOutputPath;
      if (isDirectory) {
        // 批量处理：生成输出文件名
        std::filesystem::path inputPath(imagePath);
        std::string outputFileName = inputPath.stem().string() + "_result" +
                                     inputPath.extension().string();
        finalOutputPath =
            (std::filesystem::path(outputPath) / outputFileName).string();
      } else {
        // 单张图片处理：检查outputPath是否为目录
        if (outputPath.empty()) {
          finalOutputPath = "result.jpg";
        } else if (std::filesystem::is_directory(outputPath) ||
                   outputPath.back() == '/' || outputPath.back() == '\\') {
          // outputPath是目录，生成文件名
          std::filesystem::path inputPath(imagePath);
          std::string outputFileName = inputPath.stem().string() + "_result" +
                                       inputPath.extension().string();
          finalOutputPath =
              (std::filesystem::path(outputPath) / outputFileName).string();
        } else {
          // outputPath是完整文件路径
          finalOutputPath = outputPath;
        }
      }

      if (cv::imwrite(finalOutputPath, result_image)) {
        std::cout << "Inference time: " << inference_time << "ms, "
                  << "Detections: " << drawn_count << " - " << finalOutputPath
                  << std::endl;
        success_count++;
      } else {
        std::cerr << "Error: Failed to save result to: " << finalOutputPath
                  << std::endl;
      }

      // Ensure all GPU operations complete
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaDeviceSynchronize());

      // Reset buffer states
      memset(decode_ptr_host, 0, (1 + 1000 * 7) * sizeof(float));
      CUDA_CHECK(cudaMemset(decode_ptr_device, 0, (1 + 1000 * 7) * sizeof(float)));

    } catch (const std::exception &e) {
      std::cerr << "Error processing " << imagePath << ": " << e.what()
                << std::endl;
    }

    processed_count++;
  }

  // 释放临时缓冲区
  delete[] temp_output_host;

  if (isDirectory) {
    std::cout << "Batch processing completed: " << success_count << "/"
              << processed_count << " successful" << std::endl;
  }

  // 清理资源
  if (stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  cuda_preprocess_destroy();

  if (decode_ptr_device) {
    CUDA_CHECK(cudaFree(decode_ptr_device));
  }

  if (device_buffers[0]) {
    CUDA_CHECK(cudaFree(device_buffers[0]));
  }

  if (device_buffers[1]) {
    CUDA_CHECK(cudaFree(device_buffers[1]));
  }

  if (output_buffer_host) {
    delete[] output_buffer_host;
  }

  if (decode_ptr_host) {
    delete[] decode_ptr_host;
  }

  if (context) context->destroy();
  if (engine) engine->destroy();
  if (runtime) runtime->destroy();

  CUDA_CHECK(cudaDeviceSynchronize());
}
