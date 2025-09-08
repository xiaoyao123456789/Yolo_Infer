#include "cuda_utils.hpp"
#include "preprocess.h"

// 静态缓冲区，用于存储图像数据
// 主机端内存缓冲区
static uint8_t* img_buffer_host = nullptr;
// 设备端内存缓冲区
static uint8_t* img_buffer_device = nullptr;

// CUDA核函数：执行仿射变换，将输入图像变换到目标尺寸
// 参数说明：
// src: 输入图像数据指针
// src_line_size: 输入图像每行的字节数
// src_width, src_height: 输入图像的宽度和高度
// dst: 输出图像数据指针
// dst_width, dst_height: 输出图像的宽度和高度
// const_value_st: 填充常数值
// d2s: 目标到源的仿射变换矩阵
// edge: 总处理像素数
__global__ void warpaffine_kernel(uint8_t* src, int src_line_size,
                                  int src_width, int src_height, float* dst,
                                  int dst_width, int dst_height,
                                  uint8_t const_value_st, AffineMatrix d2s,
                                  int edge) {
  // 计算当前线程处理的像素位置
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  // 如果超出边界则返回
  if (position >= edge) return;

  // 提取仿射变换矩阵的参数
  float m_x1 = d2s.value[0];
  float m_y1 = d2s.value[1];
  float m_z1 = d2s.value[2];
  float m_x2 = d2s.value[3];
  float m_y2 = d2s.value[4];
  float m_z2 = d2s.value[5];

  // 计算目标图像中的坐标(dx, dy)
  int dx = position % dst_width;
  int dy = position / dst_width;
  // 通过仿射变换计算源图像中的对应坐标(src_x, src_y)
  float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
  float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
  // 用于存储三个颜色通道的值
  float c0, c1, c2;

  if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
    // 源坐标超出图像范围，使用常数值填充
    c0 = const_value_st;
    c1 = const_value_st;
    c2 = const_value_st;
  } else {
    // 源坐标在图像范围内，进行双线性插值
    // 计算插值的四个相邻像素坐标
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    // 设置边界外的常数值
    uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
    // 计算插值权重
    float ly = src_y - y_low;
    float lx = src_x - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    // 计算四个相邻像素的权重
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    uint8_t* v1 = const_value;
    uint8_t* v2 = const_value;
    uint8_t* v3 = const_value;
    uint8_t* v4 = const_value;

    if (y_low >= 0) {
      if (x_low >= 0) v1 = src + y_low * src_line_size + x_low * 3;

      if (x_high < src_width) v2 = src + y_low * src_line_size + x_high * 3;
    }

    if (y_high < src_height) {
      if (x_low >= 0) v3 = src + y_high * src_line_size + x_low * 3;

      if (x_high < src_width) v4 = src + y_high * src_line_size + x_high * 3;
    }

    c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
    c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
    c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
  }

  // 将BGR格式转换为RGB格式（交换B和R通道）
  float t = c2;
  c2 = c0;
  c0 = t;

  // 归一化处理，将像素值从[0,255]转换到[0,1]范围
  c0 = c0 / 255.0f;
  c1 = c1 / 255.0f;
  c2 = c2 / 255.0f;

  // 将交错的RGB格式转换为平面格式(RRRGGGBBB)
  // 计算图像面积
  int area = dst_width * dst_height;
  // 计算三个通道的目标指针位置
  float* pdst_c0 = dst + dy * dst_width + dx;
  float* pdst_c1 = pdst_c0 + area;
  float* pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;
}

// CUDA预处理函数：将输入图像预处理并转换到目标尺寸
// 参数说明：
// src: 输入图像数据指针
// src_width, src_height: 输入图像的宽度和高度
// dst: 输出图像数据指针
// dst_width, dst_height: 输出图像的宽度和高度
// stream: CUDA流
void cuda_preprocess(uint8_t* src, int src_width, int src_height, float* dst,
                     int dst_width, int dst_height, cudaStream_t stream) {
  // 计算图像总字节数（3通道）
  int img_size = src_width * src_height * 3;
  // 将数据复制到固定内存（pinned memory）
  memcpy(img_buffer_host, src, img_size);
  // 将数据从主机内存异步复制到设备内存
  CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size,
                             cudaMemcpyHostToDevice, stream));

  // 定义源到目标和目标到源的仿射变换矩阵
  AffineMatrix s2d, d2s;
  // 计算缩放比例，保持宽高比
  float scale =
      std::min(dst_height / (float)src_height, dst_width / (float)src_width);

  // 设置源到目标的仿射变换矩阵参数
  // 这里实现的是缩放并居中的变换
  s2d.value[0] = scale;  // x方向缩放
  s2d.value[1] = 0;      // x方向无剪切
  s2d.value[2] =
      -scale * src_width * 0.5 + dst_width * 0.5;  // x方向平移（居中）
  s2d.value[3] = 0;                                // y方向无剪切
  s2d.value[4] = scale;                            // y方向缩放
  s2d.value[5] =
      -scale * src_height * 0.5 + dst_height * 0.5;  // y方向平移（居中）
  // 使用OpenCV矩阵表示仿射变换
  cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
  cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
  // 计算仿射变换的逆变换（从目标到源的变换）
  cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

  // 将逆变换矩阵复制到d2s结构体中
  memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

  // 计算需要处理的总像素数
  int jobs = dst_height * dst_width;
  // 设置CUDA线程块大小
  int threads = 256;
  // 计算需要的线程块数量
  int blocks = ceil(jobs / (float)threads);
  // 调用CUDA核函数执行仿射变换
  warpaffine_kernel<<<blocks, threads, 0, stream>>>(
      img_buffer_device, src_width * 3, src_width, src_height, dst, dst_width,
      dst_height, 128, d2s, jobs);
}

// 批量预处理函数：处理一批图像
// 参数说明：
// img_batch: 输入图像批次
// dst: 输出图像数据指针
// dst_width, dst_height: 输出图像的宽度和高度
// stream: CUDA流
void cuda_batch_preprocess(std::vector<cv::Mat>& img_batch, float* dst,
                           int dst_width, int dst_height, cudaStream_t stream) {
  // 计算每张目标图像的大小（字节数）
  int dst_size = dst_width * dst_height * 3;
  // 循环处理每张图像
  for (size_t i = 0; i < img_batch.size(); i++) {
    // 对每张图像调用预处理函数
    cuda_preprocess(img_batch[i].ptr(), img_batch[i].cols, img_batch[i].rows,
                    &dst[dst_size * i], dst_width, dst_height, stream);
    // 同步CUDA流，确保当前图像处理完成
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

// 初始化CUDA预处理环境
// 参数说明：
// max_image_size: 最大图像尺寸（像素数）
void cuda_preprocess_init(int max_image_size) {
  // 在固定内存（pinned memory）中分配输入数据缓冲区
  // 固定内存可以加速主机到设备的数据传输
  CUDA_CHECK(cudaMallocHost((void**)&img_buffer_host, max_image_size * 3));
  // 在设备内存中分配输入数据缓冲区
  CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, max_image_size * 3));
}

// 释放CUDA预处理环境资源
void cuda_preprocess_destroy() {
  // 释放设备内存
  CUDA_CHECK(cudaFree(img_buffer_device));
  // 释放固定内存
  CUDA_CHECK(cudaFreeHost(img_buffer_host));
}

// 保存warpaffine变换后的图像
bool save_warpaffine_image(float* dst, int dst_width, int dst_height,
                           const std::string& save_path, cudaStream_t stream) {
  try {
    // 确保参数有效
    if (dst == nullptr) {
      std::cerr << "Error: dst buffer is null" << std::endl;
      return false;
    }

    // 创建一个临时缓冲区来存储从GPU复制的数据
    float* temp_buffer = new float[dst_width * dst_height * 3];

    // 从GPU复制数据到CPU
    CUDA_CHECK(cudaMemcpyAsync(temp_buffer, dst,
                               dst_width * dst_height * 3 * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    // 同步流，确保数据复制完成
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 创建OpenCV图像
    cv::Mat output_image(dst_height, dst_width, CV_8UC3);

    // 将浮点数据转换为8位无符号整数，并从平面格式(RRRGGGBBB)转换为交错格式(BGR)
    for (int y = 0; y < dst_height; y++) {
      for (int x = 0; x < dst_width; x++) {
        int plane_offset = dst_height * dst_width;
        int pixel_offset = y * dst_width + x;

        // 确保索引在有效范围内
        if (pixel_offset < 0 ||
            pixel_offset + plane_offset * 2 >= dst_width * dst_height * 3) {
          std::cerr << "Error: Invalid pixel index" << std::endl;
          delete[] temp_buffer;
          return false;
        }

        // 获取RGB通道值（注意：在warpaffine_kernel中已经将BGR转换为RGB）
        float r = temp_buffer[pixel_offset];                     // R通道
        float g = temp_buffer[pixel_offset + plane_offset];      // G通道
        float b = temp_buffer[pixel_offset + plane_offset * 2];  // B通道

        // 将[0,1]范围转换回[0,255]范围，并转换回BGR格式（OpenCV默认格式）
        output_image.at<cv::Vec3b>(y, x)[0] =
            static_cast<uint8_t>(b * 255.0f);  // B
        output_image.at<cv::Vec3b>(y, x)[1] =
            static_cast<uint8_t>(g * 255.0f);  // G
        output_image.at<cv::Vec3b>(y, x)[2] =
            static_cast<uint8_t>(r * 255.0f);  // R
      }
    }

    // 保存图像
    bool success = cv::imwrite(save_path, output_image);

    // 释放临时缓冲区
    delete[] temp_buffer;

    return success;
  } catch (const std::exception& e) {
    std::cerr << "Error saving warpaffine image: " << e.what() << std::endl;
    return false;
  }
}