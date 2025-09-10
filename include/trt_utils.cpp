#include "trt_utils.hpp"

#include <NvInfer.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "cuda_utils.hpp"
#include "postprocess.h"
#include "preprocess.h"

using namespace nvinfer1;

// 定义全局变量
std::vector<cv::Mat> images;
// ========== 后处理相关常量定义 ==========
const static float kConfThresh =
    0.25f;  // 置信度阈值：过滤低置信度的检测结果（与Python版本保持一致）
const static float kNmsThresh =
    0.5f;  // NMS阈值：控制重叠框的抑制程度（与Python版本保持一致）
const static int kMaxNumOutputBbox =
    1000;  // 最大输出检测框数量：限制单次推理的最大检测数量
const static int kNumberOfPoints =
    17;  // YOLO关键点检测的关键点数量（如人体姿态估计）
static int kOutputSize =
    1;  // 全局变量：模型输出大小，在prepare_buffer函数中动态设置

// 定义全局日志记录器
class Logger : public ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
  }
} gLogger;

void deserialize_engine(std::string& engine_name, IRuntime** runtime,
                        ICudaEngine** engine, IExecutionContext** context) {
  // 以二进制方式打开引擎文件
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  // 获取文件大小
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  // 分配内存并读取引擎文件内容
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  // 创建TensorRT运行时、引擎和执行上下文
  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}

cv::Mat read_images(const std::string& imagePath) {
  cv::Mat image = cv::imread(imagePath);
  if (image.empty()) {
    std::cerr << "read " << imagePath << " error!" << std::endl;
    assert(false);
  }
  images.push_back(image);
  std::cout << "read " << imagePath << " success!" << std::endl;

  return image;
}

void prepare_buffer(ICudaEngine* engine, float** input_buffer_device,
                    float** output_buffer_device, float** output_buffer_host,
                    float** decode_ptr_host, float** decode_ptr_device,
                    std::string cuda_post_process) {
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. Note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);

  // 从模型中获取输出尺寸
  auto outputDims = engine->getBindingDimensions(outputIndex);
  kOutputSize = 1;  // 重置全局变量
  std::cout << "outputDims.nbDims: " << outputDims.nbDims << std::endl;
  for (int i = 0; i < outputDims.nbDims; i++) {
    std::cout << "outputDims.d[" << i << "]: " << outputDims.d[i] << std::endl;
    kOutputSize *= outputDims.d[i];
  }
  std::cout << "kOutputSize: " << kOutputSize << std::endl;

  // 动态获取模型输出维度信息来确定box_element
  int box_element = 7;  // 默认每个框7个元素：x1,y1,x2,y2,conf,cls,keep_flag
  if (outputDims.nbDims == 3) {
    box_element = outputDims.d[1];  // 获取每个框的元素数量
  }

  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)input_buffer_device,
                        kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)output_buffer_device,
                        kBatchSize * kOutputSize * sizeof(float)));

  if (cuda_post_process == "c") {
    *output_buffer_host = new float[kBatchSize * kOutputSize];
    // CPU模式下固定使用7个元素：x1,y1,x2,y2,conf,cls,keep_flag
    const int cpu_bbox_element = 7;
    *decode_ptr_host = new float[1 + kMaxNumOutputBbox * cpu_bbox_element];
    std::cout << "[DEBUG] CPU mode: allocated decode_ptr_host for "
              << (1 + kMaxNumOutputBbox * cpu_bbox_element) << " floats"
              << std::endl;
  } else if (cuda_post_process == "g") {
    if (kBatchSize > 1) {
      std::cerr << "Do not yet support GPU post processing for multiple batches"
                << std::endl;
      exit(0);
    }

    // GPU模式固定使用7个元素：x1,y1,x2,y2,conf,cls,keep_flag
    const int gpu_bbox_element = 7;

    // Allocate memory for decode_ptr_host and copy to device
    *decode_ptr_host = new float[1 + kMaxNumOutputBbox * gpu_bbox_element];
    CUDA_CHECK(
        cudaMalloc((void**)decode_ptr_device,
                   sizeof(float) * (1 + kMaxNumOutputBbox * gpu_bbox_element)));
    std::cout << "[DEBUG] GPU mode: allocated decode_ptr_host for "
              << (1 + kMaxNumOutputBbox * gpu_bbox_element) << " floats"
              << std::endl;
  }
}

/**
 * 执行模型推理和后处理的主函数
 * @param context TensorRT执行上下文，用于模型推理
 * @param stream CUDA流，用于异步执行GPU操作
 * @param buffers 输入输出缓冲区数组：[0]输入图像数据，[1]网络输出数据
 * @param output 主机端输出缓冲区（CPU后处理模式使用）
 * @param batchsize 批处理大小
 * @param decode_ptr_host 主机端解码结果缓冲区（GPU后处理模式使用）
 * @param decode_ptr_device 设备端解码结果缓冲区（GPU后处理模式使用）
 * @param model_bboxes 模型预测的边界框数量
 * @param cuda_post_process 后处理模式："c"=CPU模式，"g"=GPU模式
 */
void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers,
           float* output, int batchsize, float* decode_ptr_host,
           float* decode_ptr_device, int model_bboxes,
           std::string cuda_post_process, int img_width, int img_height) {
  // 开始计时：记录推理和后处理的总时间
  auto start = std::chrono::system_clock::now();

  // 执行TensorRT模型推理
  // buffers[0]: 输入图像数据（设备内存）
  // buffers[1]: 网络输出数据（设备内存）
  context.executeV2(buffers);

  // 分配主机端缓冲区用于存储网络原始输出
  float* raw_output_host = new float[kOutputSize];

  // 异步将网络输出从设备内存复制到主机内存
  // 这个复制操作与后续的后处理可以并行执行
  CUDA_CHECK(cudaMemcpyAsync(raw_output_host, buffers[1],
                             kOutputSize * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));

  // 等待复制完成以便保存原始输出
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // 保存原始网络输出用于调试对比
  std::ofstream raw_file("bin/cpp_raw_output.bin", std::ios::binary);
  if (raw_file.is_open()) {
    raw_file.write(reinterpret_cast<const char*>(raw_output_host),
                   kOutputSize * sizeof(float));
    raw_file.close();
    std::cout << "✅ C++ save bin/cpp_raw_output.bin" << std::endl;
    std::cout << "📌 输出 size: " << kOutputSize << ", dtype: float32"
              << std::endl;
  }

#if 1
  if (cuda_post_process == "c") {
    // CPU后处理模式：等待异步复制完成，然后在CPU上执行解码和NMS操作
    std::cout << "[DEBUG] Entering CPU post-processing mode" << std::endl;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "[DEBUG] CUDA stream synchronized" << std::endl;

    // 直接使用传入的decode_ptr_host，不再分配临时缓冲区
    const int cpu_bbox_element = 7;  // CPU后处理固定使用7个元素
    if (decode_ptr_host == nullptr) {
      std::cout << "[ERROR] decode_ptr_host is null in CPU mode!" << std::endl;
      return;
    }
    memset(decode_ptr_host, 0,
           sizeof(float) * (1 + kMaxNumOutputBbox * cpu_bbox_element));
    std::cout << "[DEBUG] Using decode_ptr_host directly, size: "
              << (1 + kMaxNumOutputBbox * cpu_bbox_element) << " floats"
              << std::endl;

    // 动态获取模型输出维度信息
    auto outputDims = context.getEngine().getBindingDimensions(1);
    int box_element = 5;  // 默认每个框5个元素：x,y,w,h,conf
    int num_class = 1;    // 默认单类别检测

    std::cout << "[DEBUG] Model output dimensions:" << std::endl;
    std::cout << "[DEBUG] outputDims.nbDims: " << outputDims.nbDims
              << std::endl;
    for (int i = 0; i < outputDims.nbDims; i++) {
      std::cout << "[DEBUG] outputDims.d[" << i << "]: " << outputDims.d[i]
                << std::endl;
    }
    std::cout << "[DEBUG] model_bboxes: " << model_bboxes << std::endl;
    std::cout << "[DEBUG] kOutputSize: " << kOutputSize << std::endl;

    // 根据输出维度确定框元素数量和类别数量
    if (outputDims.nbDims == 3) {
      box_element = outputDims.d[1];  // 获取每个框的元素数量
      // 计算类别数量：总元素数 - 4个坐标 = 置信度 + 类别数
      num_class = box_element > 5 ? (box_element - 4) : 1;
    }

    std::cout << "[CPU后处理] 开始CPU解码，输入数据维度: " << outputDims.d[0]
              << "x" << outputDims.d[1] << std::endl;
    std::cout << "[CPU后处理] box_element: " << box_element
              << ", num_class: " << num_class << std::endl;

    // 检查raw_output_host数据
    std::cout << "[DEBUG] raw_output_host first 10 values: ";
    for (int i = 0; i < std::min(10, kOutputSize); i++) {
      std::cout << raw_output_host[i] << " ";
    }
    std::cout << std::endl;

    // 检查raw_output_host是否为空或异常
    if (raw_output_host == nullptr) {
      std::cout << "[ERROR] raw_output_host is null pointer!" << std::endl;
      return;
    }

    bool has_valid_data = false;
    for (int i = 0; i < std::min(100, kOutputSize); i++) {
      if (raw_output_host[i] != 0.0f && !std::isnan(raw_output_host[i]) &&
          !std::isinf(raw_output_host[i])) {
        has_valid_data = true;
        break;
      }
    }
    std::cout << "[DEBUG] raw_output_host contains valid data: "
              << (has_valid_data ? "yes" : "no") << std::endl;

    // 调用CPU解码函数：将网络原始输出转换为检测框
    std::cout << "[DEBUG] Calling cpu_decode function with parameters:"
              << std::endl;
    std::cout << "[DEBUG] - model_bboxes: " << model_bboxes << std::endl;
    std::cout << "[DEBUG] - kConfThresh: " << kConfThresh << std::endl;
    std::cout << "[DEBUG] - kMaxNumOutputBbox: " << kMaxNumOutputBbox
              << std::endl;
    std::cout << "[DEBUG] - cpu_bbox_element: " << cpu_bbox_element
              << std::endl;
    std::cout << "[DEBUG] - num_class: " << num_class << std::endl;

    cpu_decode(raw_output_host, model_bboxes, kConfThresh, decode_ptr_host,
               kMaxNumOutputBbox, cpu_bbox_element, num_class);

    std::cout << "[DEBUG] cpu_decode function completed" << std::endl;
    std::cout << "[DEBUG] decode_ptr_host[0] (detection count): "
              << decode_ptr_host[0] << std::endl;

    // 保存解码后、NMS前的结果用于调试
    std::ofstream decode_file("bin/cpp_decode_output.bin", std::ios::binary);
    if (decode_file.is_open()) {
      decode_file.write(
          reinterpret_cast<const char*>(decode_ptr_host),
          sizeof(float) * (1 + kMaxNumOutputBbox * cpu_bbox_element));
      decode_file.close();
      std::cout << "✅ C++ 解码输出已保存到 bin/cpp_decode_output.bin"
                << std::endl;
      std::cout << "📌 解码后检测框数量: " << (int)decode_ptr_host[0]
                << std::endl;
    }

    // 在NMS前进行坐标变换（与Python版本保持一致）
    // 计算仿射逆变换矩阵
    float scale = std::min(kInputH / (float)kInputH,
                           kInputW / (float)kInputW);  // 这里需要实际图像尺寸

    // 注意：这里需要实际的图像尺寸，暂时使用固定值进行测试
    // 在实际应用中，应该从外部传入图像尺寸
    // int img_height = 1440;  // 实际图像高度
    // int img_width = 2560;   // 实际图像宽度

    scale = std::min(kInputH / (float)img_height, kInputW / (float)img_width);

    // 构建仿射变换矩阵（与preprocess.cu一致）
    cv::Mat s2d = (cv::Mat_<float>(2, 3) << scale, 0,
                   -scale * img_width * 0.5f + kInputW * 0.5f, 0, scale,
                   -scale * img_height * 0.5f + kInputH * 0.5f);

    // 计算逆变换矩阵
    cv::Mat d2s;
    cv::invertAffineTransform(s2d, d2s);
    float* m = d2s.ptr<float>(0);

    std::cout << "[CPU后处理] 应用坐标变换，scale: " << scale << std::endl;

    // 对所有检测框应用坐标变换
    int count = static_cast<int>(decode_ptr_host[0]);
    for (int i = 0; i < count; i++) {
      float* box = decode_ptr_host + 1 + i * cpu_bbox_element;

      float x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];

      // 应用仿射逆变换
      float new_x1 = m[0] * x1 + m[1] * y1 + m[2];
      float new_y1 = m[3] * x1 + m[4] * y1 + m[5];
      float new_x2 = m[0] * x2 + m[1] * y2 + m[2];
      float new_y2 = m[3] * x2 + m[4] * y2 + m[5];

      // 确保坐标在图像范围内
      new_x1 = std::max(0.0f, std::min((float)img_width, new_x1));
      new_y1 = std::max(0.0f, std::min((float)img_height, new_y1));
      new_x2 = std::max(0.0f, std::min((float)img_width, new_x2));
      new_y2 = std::max(0.0f, std::min((float)img_height, new_y2));

      // 确保x1 < x2 和 y1 < y2
      if (new_x1 > new_x2) std::swap(new_x1, new_x2);
      if (new_y1 > new_y2) std::swap(new_y1, new_y2);

      // 更新坐标
      box[0] = new_x1;
      box[1] = new_y1;
      box[2] = new_x2;
      box[3] = new_y2;
    }

    // 执行CPU版本的NMS
    cpu_nms(decode_ptr_host, kNmsThresh, kMaxNumOutputBbox);

    // 保存NMS后的最终结果用于调试
    std::ofstream nms_file("bin/cpp_nms_output.bin", std::ios::binary);
    if (nms_file.is_open()) {
      nms_file.write(
          reinterpret_cast<const char*>(decode_ptr_host),
          sizeof(float) * (1 + kMaxNumOutputBbox * cpu_bbox_element));
      nms_file.close();
      std::cout << "✅ C++ NMS输出已保存到 bin/cpp_nms_output.bin" << std::endl;
      std::cout << "📌 NMS后检测框数量: " << (int)decode_ptr_host[0]
                << std::endl;

      // 打印前几个检测框的详细信息
      int count = std::min((int)decode_ptr_host[0], 5);
      for (int i = 0; i < count; i++) {
        float* box = decode_ptr_host + 1 + i * cpu_bbox_element;
        std::cout << "框 " << i << ": [" << box[0] << ", " << box[1] << ", "
                  << box[2] << ", " << box[3] << "], 置信度: " << box[4]
                  << ", 类别: " << (int)box[5] << ", keep: " << (int)box[6]
                  << std::endl;
      }
    }

    // CPU后处理结果已经直接存储在decode_ptr_host中，无需复制
    std::cout << "[DEBUG] CPU processing completed, detection count: "
              << decode_ptr_host[0] << std::endl;

    if (output != nullptr) {
      memcpy(output, raw_output_host, batchsize * kOutputSize * sizeof(float));
    }

    // 计算并输出推理和CPU后处理时间
    auto end = std::chrono::system_clock::now();
    std::cout << "inference and cpu postprocess time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;
  } else if (cuda_post_process == "g") {
    // GPU后处理模式：在GPU上执行解码和NMS操作

    // 动态获取模型输出维度信息
    // 不同的YOLO模型可能有不同的输出格式
    auto outputDims = context.getEngine().getBindingDimensions(1);
    int box_element = 5;  // 默认每个框5个元素：x,y,w,h,conf
    int num_class = 1;    // 默认单类别检测

    // 根据输出维度确定框元素数量和类别数量
    if (outputDims.nbDims == 3) {
      box_element = outputDims.d[1];  // 获取每个框的元素数量
      // 计算类别数量：总元素数 - 4个坐标 = 置信度 + 类别数
      num_class = box_element > 5 ? (box_element - 4) : 1;
    }

    // GPU后处理固定使用7个元素：x1,y1,x2,y2,conf,cls,keep_flag
    const int gpu_bbox_element = 7;

    // 初始化解码输出缓冲区为0
    // decode_ptr_device用于存储解码后的检测结果
    // 缓冲区大小：1个计数器 + 最大检测框数量 * 每个框的元素数量
    CUDA_CHECK(cudaMemsetAsync(
        decode_ptr_device, 0,
        sizeof(float) * (1 + kMaxNumOutputBbox * gpu_bbox_element), stream));

    // 调用CUDA解码函数：将网络原始输出转换为检测框
    std::cout << "[DEBUG] 开始GPU解码，model_bboxes=" << model_bboxes
              << ", kMaxNumOutputBbox=" << kMaxNumOutputBbox
              << ", gpu_bbox_element=" << gpu_bbox_element << std::endl;

    cuda_decode((float*)buffers[1], model_bboxes, kConfThresh,
                decode_ptr_device, kMaxNumOutputBbox, stream, gpu_bbox_element,
                num_class);

    // 检查解码后的CUDA错误
    cudaError_t decode_error = cudaGetLastError();
    if (decode_error != cudaSuccess) {
      std::cout << "[ERROR] GPU解码失败: " << cudaGetErrorString(decode_error)
                << std::endl;
    } else {
      std::cout << "[DEBUG] GPU解码完成" << std::endl;
    }

    // 保存解码后、NMS前的结果用于调试
    float* decode_before_nms =
        new float[1 + kMaxNumOutputBbox * gpu_bbox_element];
    CUDA_CHECK(cudaMemcpyAsync(
        decode_before_nms, decode_ptr_device,
        sizeof(float) * (1 + kMaxNumOutputBbox * gpu_bbox_element),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::ofstream decode_file("bin/cpp_decode_output.bin", std::ios::binary);
    if (decode_file.is_open()) {
      decode_file.write(
          reinterpret_cast<const char*>(decode_before_nms),
          sizeof(float) * (1 + kMaxNumOutputBbox * gpu_bbox_element));
      decode_file.close();
      std::cout << "✅ C++ 解码输出已保存到 bin/cpp_decode_output.bin"
                << std::endl;
      std::cout << "📌 解码后检测框数量: " << (int)decode_before_nms[0]
                << std::endl;
    }

    // GPU模式也需要进行坐标变换（与CPU模式保持一致）
    // 计算仿射逆变换矩阵
    // int img_height = 1440;  // 实际图像高度
    // int img_width = 2560;   // 实际图像宽度
    float scale =
        std::min(kInputH / (float)img_height, kInputW / (float)img_width);
    float dx = (kInputW - scale * img_width) / 2;
    float dy = (kInputH - scale * img_height) / 2;

    std::cout << "[DEBUG] GPU坐标变换参数: scale=" << scale << ", dx=" << dx
              << ", dy=" << dy << std::endl;

    // 对解码后的坐标进行变换
    int count = (int)decode_before_nms[0];
    for (int i = 0; i < count; i++) {
      float* box = decode_before_nms + 1 + i * gpu_bbox_element;
      float x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];

      // 逆变换：从模型坐标系转换回原图坐标系
      float new_x1 = (x1 - dx) / scale;
      float new_y1 = (y1 - dy) / scale;
      float new_x2 = (x2 - dx) / scale;
      float new_y2 = (y2 - dy) / scale;

      box[0] = new_x1;
      box[1] = new_y1;
      box[2] = new_x2;
      box[3] = new_y2;
    }

    // 将变换后的坐标复制回GPU内存
    CUDA_CHECK(cudaMemcpyAsync(
        decode_ptr_device, decode_before_nms,
        sizeof(float) * (1 + kMaxNumOutputBbox * gpu_bbox_element),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    delete[] decode_before_nms;

    // 执行非极大值抑制（NMS）后处理
    std::cout << "[DEBUG] 开始GPU NMS" << std::endl;

    cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);

    // 检查NMS后的CUDA错误
    cudaError_t nms_error = cudaGetLastError();
    if (nms_error != cudaSuccess) {
      std::cout << "[ERROR] GPU NMS失败: " << cudaGetErrorString(nms_error)
                << std::endl;
    } else {
      std::cout << "[DEBUG] GPU NMS完成" << std::endl;
    }

    // 将GPU处理后的结果复制回主机内存
    // 复制内容：检测框计数器 + 所有检测框数据
    // decode_ptr_host: 主机端缓冲区，用于存储最终的检测结果
    // decode_ptr_device: 设备端缓冲区，包含NMS处理后的结果
    CUDA_CHECK(cudaMemcpyAsync(
        decode_ptr_host, decode_ptr_device,
        sizeof(float) * (1 + kMaxNumOutputBbox * gpu_bbox_element),
        cudaMemcpyDeviceToHost, stream));

    // 同步流以确保所有数据复制完成
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 保存NMS后的最终结果用于调试
    std::ofstream nms_file("bin/cpp_nms_output.bin", std::ios::binary);
    if (nms_file.is_open()) {
      nms_file.write(
          reinterpret_cast<const char*>(decode_ptr_host),
          sizeof(float) * (1 + kMaxNumOutputBbox * gpu_bbox_element));
      nms_file.close();
      std::cout << "✅ C++ NMS输出已保存到 bin/cpp_nms_output.bin" << std::endl;

      // 添加边界检查，防止decode_ptr_host访问越界
      if (decode_ptr_host != nullptr) {
        int detection_count = (int)decode_ptr_host[0];
        // 确保检测框数量在合理范围内
        if (detection_count >= 0 && detection_count <= kMaxNumOutputBbox) {
          std::cout << "📌 NMS后检测框数量: " << detection_count << std::endl;

          // 打印前几个检测框的详细信息
          int count = std::min(detection_count, 5);
          for (int i = 0; i < count; i++) {
            float* box = decode_ptr_host + 1 + i * gpu_bbox_element;
            std::cout << "框 " << i << ": [" << box[0] << ", " << box[1] << ", "
                      << box[2] << ", " << box[3] << "], 置信度: " << box[4]
                      << ", 类别: " << (int)box[5] << ", keep: " << (int)box[6]
                      << std::endl;
          }
        } else {
          std::cout << "⚠️ 警告: 检测框数量异常: " << detection_count
                    << ", 可能存在内存访问错误" << std::endl;
        }
      } else {
        std::cout << "⚠️ 错误: decode_ptr_host为空指针" << std::endl;
      }
    }

    if (output != nullptr) {
      memcpy(output, raw_output_host, batchsize * kOutputSize * sizeof(float));
    }

    // 计算并输出总的推理和GPU后处理时间
    auto end = std::chrono::system_clock::now();
    std::cout << "inference and gpu postprocess time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;
  }
#endif

  // 释放临时缓冲区
  delete[] raw_output_host;

  // GPU模式和CPU模式都已经在各自的分支中同步过了
}