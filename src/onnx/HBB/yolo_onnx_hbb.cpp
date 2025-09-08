#include "yolo_onnx_hbb.hpp"

#include <iostream>
#include <filesystem>
#include <algorithm>

#include "utils.hpp"

float computeIoU_HBB(const cv::Rect& box1, const cv::Rect& box2) {
  int x1 = std::max(box1.x, box2.x);
  int y1 = std::max(box1.y, box2.y);
  int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
  int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

  int w = std::max(0, x2 - x1);
  int h = std::max(0, y2 - y1);

  float inter = w * h;
  float area1 = box1.area();
  float area2 = box2.area();
  return inter / (area1 + area2 - inter);
}

void hbb_nms(std::vector<cv::Rect>& boxes, std::vector<float>& scores,
             float score_threshold, float nms_threshold,
             std::vector<int>& indices) {
  struct BoxScore {
    cv::Rect box;
    float score;
    int id;
  };

  std::vector<BoxScore> box_scores;
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (scores[i] > score_threshold)
      box_scores.push_back({boxes[i], scores[i], (int)i});
  }

  std::sort(
      box_scores.begin(), box_scores.end(),
      [](const BoxScore& a, const BoxScore& b) { return a.score > b.score; });

  std::vector<bool> suppressed(box_scores.size(), false);

  for (size_t i = 0; i < box_scores.size(); ++i) {
    if (suppressed[i]) continue;
    for (size_t j = i + 1; j < box_scores.size(); ++j) {
      if (suppressed[j]) continue;
      float iou = computeIoU_HBB(box_scores[i].box, box_scores[j].box);
      if (iou >= nms_threshold) suppressed[j] = true;
    }
  }

  for (size_t i = 0; i < box_scores.size(); ++i) {
    if (!suppressed[i]) indices.push_back(box_scores[i].id);
  }
}

void draw_result_hbb(cv::Mat& img, const std::string& label,
                     const cv::Rect& box) {
  cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);
  cv::putText(img, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(255, 0, 0), 2);
}

void scale_box_rect(cv::Rect& box, cv::Size orig_size) {
  std::cout << "scale_box_rect: " << box << "\n";
  float g = std::min(input_width * 1.f / orig_size.width,
                     input_height * 1.f / orig_size.height);
  int pad_w = static_cast<int>((input_width - orig_size.width * g) / 2);
  int pad_h = static_cast<int>((input_height - orig_size.height * g) / 2);

  box.x = static_cast<int>((box.x - pad_w) / g);
  box.y = static_cast<int>((box.y - pad_h) / g);
  box.width = static_cast<int>(box.width / g);
  box.height = static_cast<int>(box.height / g);
}

void post_process_hbb(cv::Mat& origin, cv::Mat& result,
                      std::vector<Ort::Value>& outputs) {
  if (outputs.empty() || !outputs[0].IsTensor()) {
    std::cerr << "Invalid output tensor\n";
    return;
  }
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> cls_ids;

  const float* out = outputs[0].GetTensorData<float>();
  auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  int num_box = shape[1];
  int feat_dim = shape[2];
  int num_class = shape[2] - 4;  // 类别数
  std::cout << "num_box: " << num_box << "\n";
  std::cout << "feat_dim: " << feat_dim << "\n";

  for (int i = 0; i < num_box; ++i) {
    const float* ptr = out + i * feat_dim;
    int label = 0;
    float max_conf = ptr[4];

    for (int c = 1; c < num_class; ++c) {
      float conf = ptr[4 + c];
      if (conf > max_conf) {
        max_conf = conf;
        label = c;
      }
    }
    if (max_conf < score_threshold) continue;

    float cx = ptr[0], cy = ptr[1], w = ptr[2], h = ptr[3];
    std::cout << "max_conf: " << max_conf << "\n";
    std::cout << "cx: " << cx << "\n";
    std::cout << "cy: " << cy << "\n";
    std::cout << "w: " << w << "\n";
    std::cout << "h: " << h << "\n";

    cv::Rect rect(cx, cy, w, h);

    // 3. 映射回原图尺寸
    scale_box_rect(rect, origin.size());

    boxes.push_back(rect);
    scores.push_back(max_conf);
    cls_ids.push_back(label);
  }

  //   std::vector<int> keep;
  //   //   std::cout << "boxes: " << boxes.size() << "\n";
  //   hbb_nms(boxes, scores, score_threshold, nms_threshold, keep);

  //   for (int idx : keep) {
  //     std::string label =
  //         class_names[cls_ids[idx]] + ":" + cv::format("%.2f", scores[idx]);
  //     draw_result_hbb(result, label, boxes[idx]);
  //   }
}

void main_run_hbb(const std::string& imagePath, const std::wstring& modelPath,
                  const std::string& outputPath) {
  cv::Mat image = cv::imread(imagePath);
  if (image.empty()) {
    std::cerr << "Failed to read image: " << imagePath << "\n";
    return;
  }

  std::vector<float> inputs;
  pre_process(image, inputs);

  std::vector<Ort::Value> outputs;
  process(modelPath.c_str(), inputs, outputs);

  cv::Mat result = image.clone();
  post_process_hbb(image, result, outputs);
  cv::imwrite(outputPath, result);
}

// ========================== BatchHBBProcessor 实现 ==========================

/**
 * 构造函数：初始化批量处理器并加载模型
 * @param modelPath 模型文件路径（支持ONNX格式）
 * 注意：模型只在这里加载一次，后续所有图片都复用这个会话
 */
BatchHBBProcessor::BatchHBBProcessor(const std::wstring& modelPath) {
  try {
    // 创建ONNX Runtime会话，这是最耗时的操作，只执行一次
    session_ = std::make_unique<OrtSessionWrapper>(modelPath.c_str());
    std::cout << "模型加载成功: " << std::string(modelPath.begin(), modelPath.end()) << "\n";
  } catch (const std::exception& e) {
    std::cerr << "模型加载失败: " << e.what() << "\n";
    throw;  // 重新抛出异常，让调用者知道初始化失败
  }
}

/**
 * 析构函数：自动清理资源
 */
BatchHBBProcessor::~BatchHBBProcessor() = default;

/**
 * 处理单张图片的核心函数
 * @param imagePath 输入图片路径
 * @param outputPath 输出图片路径
 * @return 处理成功返回true，失败返回false
 */
bool BatchHBBProcessor::processImage(const std::string& imagePath, const std::string& outputPath) {
  try {
    // 1. 读取图片
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
      std::cerr << "无法读取图片: " << imagePath << "\n";
      return false;
    }

    // 2. 图片预处理（缩放、归一化等）
    std::vector<float> inputs;
    pre_process(image, inputs);

    // 3. 模型推理（使用已加载的会话，无需重新加载模型）
    std::vector<Ort::Value> outputs;
    session_->run_inference(inputs, outputs);

    // 4. 后处理（解析检测结果，绘制边界框）
    cv::Mat result = image.clone();
    post_process_hbb(image, result, outputs);
    
    // 5. 保存结果图片
    if (!cv::imwrite(outputPath, result)) {
      std::cerr << "无法保存结果图片: " << outputPath << "\n";
      return false;
    }
    
    return true;
  } catch (const std::exception& e) {
    std::cerr << "处理图片时出错 " << imagePath << ": " << e.what() << "\n";
    return false;
  }
}

/**
 * 扫描目录，获取所有支持的图片文件
 * @param dirPath 目录路径
 * @return 图片文件路径列表（已排序）
 */
std::vector<std::string> BatchHBBProcessor::getSupportedImageFiles(const std::string& dirPath) {
  std::vector<std::string> imageFiles;
  // 支持的图片格式列表
  const std::vector<std::string> supportedExts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
  
  try {
    // 遍历目录中的所有文件
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
      if (entry.is_regular_file()) {  // 只处理普通文件，跳过子目录
        // 获取文件扩展名并转为小写
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        // 检查是否为支持的图片格式
        if (std::find(supportedExts.begin(), supportedExts.end(), ext) != supportedExts.end()) {
          imageFiles.push_back(entry.path().string());
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "读取目录出错 " << dirPath << ": " << e.what() << "\n";
  }
  
  // 按文件名排序，确保处理顺序一致
  std::sort(imageFiles.begin(), imageFiles.end());
  return imageFiles;
}

/**
 * 生成输出文件路径
 * @param inputPath 输入文件路径
 * @param outputDir 输出目录
 * @return 输出文件的完整路径（原文件名 + "_result" + 原扩展名）
 * 例如：input.jpg -> output_dir/input_result.jpg
 */
std::string BatchHBBProcessor::generateOutputPath(const std::string& inputPath, const std::string& outputDir) {
  std::filesystem::path inputFilePath(inputPath);
  // 构造输出文件名：原文件名 + "_result" + 原扩展名
  std::string filename = inputFilePath.stem().string() + "_result" + inputFilePath.extension().string();
  return (std::filesystem::path(outputDir) / filename).string();
}

/**
 * 批量处理主函数
 * @param inputPath 输入路径（可以是单个文件或目录）
 * @param outputDir 输出目录
 * @return 处理成功返回true，失败返回false
 */
bool BatchHBBProcessor::batchProcess(const std::string& inputPath, const std::string& outputDir) {
  // 1. 确保输出目录存在（如果不存在则创建）
  try {
    std::filesystem::create_directories(outputDir);
  } catch (const std::exception& e) {
    std::cerr << "无法创建输出目录: " << e.what() << "\n";
    return false;
  }
  
  std::vector<std::string> imageFiles;
  
  // 2. 判断输入是单个文件还是目录
  if (std::filesystem::is_regular_file(inputPath)) {
    // 单个文件：直接添加到处理列表
    imageFiles.push_back(inputPath);
  } else if (std::filesystem::is_directory(inputPath)) {
    // 目录：扫描所有支持的图片文件
    imageFiles = getSupportedImageFiles(inputPath);
  } else {
    std::cerr << "无效的输入路径: " << inputPath << "\n";
    return false;
  }
  
  // 3. 检查是否找到了图片文件
  if (imageFiles.empty()) {
    std::cerr << "在路径中未找到支持的图片文件: " << inputPath << "\n";
    return false;
  }
  
  std::cout << "找到 " << imageFiles.size() << " 张图片待处理\n";
  
  // 4. 逐个处理图片，显示进度
  int successCount = 0;  // 成功处理的图片数量
  int totalCount = static_cast<int>(imageFiles.size());
  
  for (int i = 0; i < totalCount; ++i) {
    const std::string& imagePath = imageFiles[i];
    std::string outputPath = generateOutputPath(imagePath, outputDir);
    
    // 显示当前处理进度
    std::cout << "正在处理 [" << (i + 1) << "/" << totalCount << "]: " 
              << std::filesystem::path(imagePath).filename().string() << "...";
    
    // 处理单张图片
    if (processImage(imagePath, outputPath)) {
      successCount++;
      std::cout << " ✓\n";  // 成功标记
    } else {
      std::cout << " ✗\n";  // 失败标记
    }
  }
  
  // 5. 显示最终统计结果
  std::cout << "\n批量处理完成: " << successCount << "/" << totalCount << " 张图片处理成功\n";
  return successCount > 0;  // 只要有一张图片处理成功就返回true
}

// ========================== 批量处理函数 ==========================

/**
 * 批量处理的简化接口函数
 * @param inputPath 输入路径（文件或目录）
 * @param modelPath 模型文件路径
 * @param outputDir 输出目录
 * 
 * 使用说明：
 * 1. 这个函数会自动创建BatchHBBProcessor实例
 * 2. 模型只加载一次，然后处理所有图片
 * 3. 相比原来的main_run_hbb，这个函数效率更高
 */
void batch_run_hbb(const std::string& inputPath, const std::wstring& modelPath,
                   const std::string& outputDir) {
  try {
    // 创建批量处理器（模型在这里加载）
    BatchHBBProcessor processor(modelPath);
    // 执行批量处理
    processor.batchProcess(inputPath, outputDir);
  } catch (const std::exception& e) {
    std::cerr << "批量处理失败: " << e.what() << "\n";
  }
}
