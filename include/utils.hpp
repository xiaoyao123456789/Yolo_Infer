
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>  // 补充vector头文件

inline const std::vector<std::string> class_names = {"Crack", "Patch"};
inline const int input_width = 1280;
inline const int input_height = 1280;
// inline const int num_classes = static_cast<int>(class_names.size());
inline const float score_threshold = 0.25f;
inline const float nms_threshold = 0.5f;

// =========================== OrtSession 包装类 =============================
class OrtSessionWrapper {
 public:
  static Ort::Env& get_env();                               // 单例 Env
  explicit OrtSessionWrapper(const wchar_t* model_path);    // 构造
  void run_inference(const std::vector<float>& input_data,  // 推理接口
                     std::vector<Ort::Value>& outputs);

 private:
  void setup_io_info();  // 解析 I/O

  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_names_ptrs_;
  std::vector<const char*> output_names_ptrs_;
  std::vector<int64_t> input_dims_;
};

// ========================== 功能函数声明 ====================================
void process(const wchar_t* model, std::vector<float>& inputs,
             std::vector<Ort::Value>& outputs);

// 添加cv和std命名空间限定（根据实现文件中的实际使用）
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
               const cv::Size& newShape = cv::Size(1280, 1280),
               const cv::Scalar& color = cv::Scalar(114, 114, 114));

cv::Vec4d Seg_LetterBox(const cv::Mat& image, cv::Mat& outImage,
                        const cv::Size& newShape, const cv::Scalar& color);

void pre_process(cv::Mat& image, std::vector<float>& inputs);
cv::Vec4d Seg_pre_process(cv::Mat& image, std::vector<float>& inputs);