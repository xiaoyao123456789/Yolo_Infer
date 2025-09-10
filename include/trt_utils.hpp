#include <NvInfer.h>

#include <cassert>  // 用于assert
#include <filesystem>
#include <fstream>  // 用于std::ifstream
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace nvinfer1;

// 模型相关常量定义
static const char* kInputTensorName = "images";
static const char* kOutputTensorName = "output0";
static const int kBatchSize = 1;
static const int kInputH = 1280;
static const int kInputW = 1280;
// kOutputSize 和 kMaxNumOutputBbox 将从模型中动态获取
static const int bbox_element =
    7;  // GPU后处理输出格式: [x1, y1, x2, y2, conf, cls, keep_flag]

// Logger类的前向声明
class Logger;
extern Logger gLogger;

extern std::vector<cv::Mat> images;

void deserialize_engine(std::string& engine_name, IRuntime** runtime,
                        ICudaEngine** engine, IExecutionContext** context);

cv::Mat read_images(const std::string& imagePath);

void prepare_buffer(ICudaEngine* engine, float** input_buffer_device,
                    float** output_buffer_device, float** output_buffer_host,
                    float** decode_ptr_host, float** decode_ptr_device,
                    std::string cuda_post_process);

void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers,
           float* output, int batchsize, float* decode_ptr_host,
           float* decode_ptr_device, int model_bboxes,
           std::string cuda_post_process, int img_width = 0, int img_height = 0);