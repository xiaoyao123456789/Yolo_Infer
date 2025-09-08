#pragma once

#include <map>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

// TensorRT includes - conditional compilation
#ifdef TENSORRT_AVAILABLE
#include "NvInfer.h"
#endif

const static int max_image_size = 3000 * 3000;

struct AffineMatrix {
  float value[6];
};

void cuda_preprocess_init(int max_image_size);

void cuda_preprocess_destroy();

void cuda_preprocess(uint8_t* src, int src_width, int src_height, float* dst,
                     int dst_width, int dst_height, cudaStream_t stream);

void cuda_batch_preprocess(std::vector<cv::Mat>& img_batch, float* dst,
                           int dst_width, int dst_height, cudaStream_t stream);

// 保存warpaffine变换后的图像
bool save_warpaffine_image(float* dst, int dst_width, int dst_height,
                           const std::string& save_path, cudaStream_t stream);