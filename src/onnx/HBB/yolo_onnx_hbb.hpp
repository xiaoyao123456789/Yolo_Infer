// HBB = [x, y, w, h, max_class_prob, class_index]
#pragma once
#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>

void hbb_nms(std::vector<cv::Rect>& boxes, std::vector<float>& scores,
             float score_threshold, float nms_threshold,
             std::vector<int>& indices);

void draw_result_hbb(cv::Mat& img, const std::string& label,
                     const cv::Rect& box);

void post_process_hbb(cv::Mat& origin, cv::Mat& result,
                      std::vector<Ort::Value>& outputs);

void main_run_hbb(const std::string& imagePath, const std::wstring& modelPath,
                  const std::string& outputPath);

// 批量处理类
class BatchHBBProcessor {
public:
    explicit BatchHBBProcessor(const std::wstring& modelPath);
    ~BatchHBBProcessor();
    
    // 处理单张图片
    bool processImage(const std::string& imagePath, const std::string& outputPath);
    
    // 批量处理
    bool batchProcess(const std::string& inputPath, const std::string& outputDir);
    
private:
    std::unique_ptr<class OrtSessionWrapper> session_;
    std::vector<std::string> getSupportedImageFiles(const std::string& dirPath);
    std::string generateOutputPath(const std::string& inputPath, const std::string& outputDir);
};

// 批量处理函数
void batch_run_hbb(const std::string& inputPath, const std::wstring& modelPath,
                   const std::string& outputDir);
