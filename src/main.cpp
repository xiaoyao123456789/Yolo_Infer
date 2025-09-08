#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>


// #include "yolo_onnx_hbb.hpp"
// #include "yolo_onnx_obb.hpp"
// #include "yolo_onnx_seg.hpp"

#include "yolo_trt_hbb.hpp"
// #include "yolo_trt_obb.hpp"
// #include "yolo_trt_seg.hpp"

int main() {
  std::string imagePath =
      "D:/Yy/Projects/Yolo_Infer/build/windows/x64/releasedbg/"
      "2025042416192465632.jpg";
  //   std::wstring modelPath =
  //       L"D:/Yy/Projects/Yolo_Infer/build/windows/x64/releasedbg/"
  //       L"Crack_Only2_20250415.onnx";  // 确保你的模型文件路径正确

  std::wstring modelPath =
      L"D:/Yy/Projects/Yolo_Infer/build/windows/x64/releasedbg/"
      L"Abandoned_V7_20250625.trtmodel";
  std::string outputPath = "result.jpg";

  maintrt_run_hbb(imagePath, modelPath, outputPath);
  return 0;
}
