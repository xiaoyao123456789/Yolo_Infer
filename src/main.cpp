#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

// #include "yolo_onnx_hbb.hpp"
// #include "yolo_onnx_obb.hpp"
// #include "yolo_onnx_seg.hpp"

#include "yolo_trt_hbb.hpp"
// #include "yolo_trt_obb.hpp"
// #include "yolo_trt_seg.hpp"

int main(int argc, char* argv[]) {
  // 可以指定单张图片或目录路径进行批量处理
  std::string inputPath;
  if (argc > 1) {
    inputPath = argv[1];  // 使用命令行参数指定的路径
  } else {
    inputPath = "D:/Yy/Projects/Yolo_Infer/input";  // 默认目录路径用于批量处理
  }

  //   std::wstring modelPath =
  //       L"D:/Yy/Projects/Yolo_Infer/build/windows/x64/releasedbg/"
  //       L"Crack_Only2_20250415.onnx";  // 确保你的模型文件路径正确

  std::wstring modelPath =
      L"D:/Yy/Projects/Yolo_Infer/build/windows/x64/releasedbg/"
      L"Abandoned_V7_20250625.trtmodel";
  std::string outputPath = "D:/Yy/Projects/Yolo_Infer/output/";

  maintrt_run_hbb(inputPath, modelPath, outputPath);
  return 0;
}
