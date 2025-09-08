
#ifndef YOLO_TRT_HBB_HPP
#define YOLO_TRT_HBB_HPP

#include <string>

// Forward declarations to avoid including heavy headers in header file
namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

void maintrt_run_hbb(const std::string& imagePath,
                     const std::wstring& modelPath,
                     const std::string& outputPath);

#endif