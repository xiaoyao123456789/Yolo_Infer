
#include <iostream>
#include <opencv2/opencv.hpp>

#include "opencv2/highgui.hpp"
#include "yolo_onnx_obb.hpp"

int main() {
  std::string imagePath = "D:/Yy/Pictures/Camera Roll/2024091817123697118.jpg";
  std::wstring modelPath =
      L"D:/Yy/Projects/Yolo_Infer/build/windows/x64/releasedbg/"
      L"Crack_Only2_20250415.onnx";  // 确保你的模型文件路径正确
  std::string outputPath = "result.jpg";

  main_run(imagePath, modelPath, outputPath);
  return 0;
}
