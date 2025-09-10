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

  return image;
}

/**
 * @brief 准备CUDA推理所需的缓冲区（包括主机和设备内存）
 *
 * 该函数根据TensorRT引擎的绑定信息，为输入、输出以及后处理相关数据分配GPU和CPU内存。
 * 支持两种后处理模式：CPU模式（'c'）和GPU模式（'g'），并根据不同模式分配相应的解码缓冲区。
 *
 * @param engine TensorRT引擎指针，用于获取绑定信息和张量维度
 * @param input_buffer_device 输入张量的GPU设备缓冲区指针
 * @param output_buffer_device 输出张量的GPU设备缓冲区指针
 * @param output_buffer_host 输出张量的CPU主机缓冲区指针（主要用于CPU后处理）
 * @param decode_ptr_host 解码结果在CPU上的存储缓冲区（存放最终检测框等信息）
 * @param decode_ptr_device 解码结果在GPU上的存储缓冲区（用于GPU后处理）
 * @param cuda_post_process 后处理方式："c"表示CPU处理，"g"表示GPU处理
 */
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
  for (int i = 0; i < outputDims.nbDims; i++) {
    kOutputSize *= outputDims.d[i];
  }

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
    // 注册为pinned内存以提升潜在的D2H性能
    CUDA_CHECK(cudaHostRegister(
        *decode_ptr_host,
        sizeof(float) * (1 + kMaxNumOutputBbox * cpu_bbox_element),
        cudaHostRegisterDefault));

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
    // 注册为pinned内存以提升D2H性能
    CUDA_CHECK(cudaHostRegister(
        *decode_ptr_host,
        sizeof(float) * (1 + kMaxNumOutputBbox * gpu_bbox_element),
        cudaHostRegisterDefault));
    CUDA_CHECK(
        cudaMalloc((void**)decode_ptr_device,
                   sizeof(float) * (1 + kMaxNumOutputBbox * gpu_bbox_element)));
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
  // 创建CUDA事件用于精确测量GPU推理时间
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // 记录开始事件
  CUDA_CHECK(cudaEventRecord(start, stream));

  // 执行TensorRT模型推理
  // buffers[0]: 输入图像数据（device）
  // buffers[1]: 网络输出数据（device）
  bool trt_ok = context.enqueueV2(buffers, stream, nullptr);
  if (!trt_ok) {
    std::cerr << "[TensorRT] enqueueV2 failed" << std::endl;
  }

  // 此时 raw_output_host 里就是模型推理结果，可以直接进入后处理
  float* raw_output_host = new float[kOutputSize];

  // 记录结束事件并同步
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // 计算推理时间
  float elapsed_ms;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  std::cout << "detect" << elapsed_ms << " ms" << std::endl;

  // 销毁事件
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

#if 1
  if (cuda_post_process == "c") {
    // CPU后处理模式：等待异步复制完成，然后在CPU上执行解码和NMS操作
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (decode_ptr_host == nullptr) {
      std::cout << "[ERROR] decode_ptr_host is null in CPU mode!" << std::endl;
      return;
    }

    // 动态获取模型输出维度信息
    auto outputDims = context.getEngine().getBindingDimensions(1);
    int box_element = 5;  // 默认每个框5个元素：x,y,w,h,conf
    int num_class = 1;    // 默认单类别检测

    // 根据输出维度确定框元素数量和类别数量
    if (outputDims.nbDims == 3) {
      box_element = outputDims.d[1];  // 获取每个框的元素数量
      // 计算类别数量：总元素数 - 4个坐标 = 置信度 + 类别数
      num_class = box_element > 5 ? (box_element - 4) : 1;
    }

    // 调用CPU解码函数：将网络原始输出转换为检测框
    cpu_decode(raw_output_host, model_bboxes, kConfThresh, decode_ptr_host,
               kMaxNumOutputBbox, bbox_element, num_class);

    // 在NMS前进行坐标变换（与Python版本保持一致）
    // 计算仿射逆变换矩阵
    float scale =
        std::min(kInputH / (float)img_height, kInputW / (float)img_width);

    // 构建仿射变换矩阵（与preprocess.cu一致）
    cv::Mat s2d = (cv::Mat_<float>(2, 3) << scale, 0,
                   -scale * img_width * 0.5f + kInputW * 0.5f, 0, scale,
                   -scale * img_height * 0.5f + kInputH * 0.5f);

    // 计算逆变换矩阵
    cv::Mat d2s;
    cv::invertAffineTransform(s2d, d2s);
    float* m = d2s.ptr<float>(0);

    // 对所有检测框应用坐标变换
    int count = static_cast<int>(decode_ptr_host[0]);
    for (int i = 0; i < count; i++) {
      float* box = decode_ptr_host + 1 + i * bbox_element;

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
      nms_file.write(reinterpret_cast<const char*>(decode_ptr_host),
                     sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element));
      nms_file.close();
    }

    // CPU后处理结果已经直接存储在decode_ptr_host中，无需复制
    if (output != nullptr) {
      memcpy(output, raw_output_host, batchsize * kOutputSize * sizeof(float));
    }

    // ===== 验证缓冲区清零效果 =====
    // Verify clearing effect
    int drawn_count = static_cast<int>(decode_ptr_host[0]);

    // 注意：这里的时间测量已经在函数开始处使用CUDA事件完成
    // CPU后处理时间可以单独测量，但推理时间已经通过CUDA事件记录
  } else if (cuda_post_process == "g") {
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

    // 计数器清零
    CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float), stream));

    // 调用CUDA解码函数：将网络原始输出转换为检测框
    cuda_decode((float*)buffers[1], model_bboxes, kConfThresh,
                decode_ptr_device, kMaxNumOutputBbox, stream, gpu_bbox_element,
                num_class);

    // GPU模式：直接在GPU上进行坐标变换，无需CPU处理
    // 调用GPU坐标变换核函数，避免GPU->CPU->GPU的数据传输
    cuda_coordinate_transform(decode_ptr_device, img_width, img_height, kInputW,
                              kInputH, kMaxNumOutputBbox, stream);

    // 执行非极大值抑制（NMS）后处理
    cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);

    // 将结果从设备内存复制到主机内存
    CUDA_CHECK(cudaMemcpyAsync(
        decode_ptr_host, decode_ptr_device,
        sizeof(float) * (1 + kMaxNumOutputBbox * gpu_bbox_element),
        cudaMemcpyDeviceToHost, stream));

    // 同步流以确保所有数据复制完成
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 注意：推理时间已经在函数开始处使用CUDA事件记录
    // GPU后处理时间包含在整个流程中，可以通过CUDA事件单独测量
  }
#endif

  // raw_output_host是std::vector，会自动释放内存

  // GPU模式和CPU模式都已经在各自的分支中同步过了
}
