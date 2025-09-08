import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import cv2
import os
import random
from cv2.dnn import NMSBoxes

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path, engine_file_path):
    if os.path.exists(engine_file_path):
        return
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 28
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        opt_shape = [1, 3, 1280, 1280]
        profile.set_shape(input_tensor.name, opt_shape, opt_shape, opt_shape)
        config.add_optimization_profile(profile)
        engine = builder.build_engine(network, config)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def cuda_like_warpaffine(img, d2s_matrix, dst_width, dst_height):
    """实现与CUDA核函数完全一致的仿射变换"""
    src_height, src_width = img.shape[:2]
    dst = np.zeros((dst_height, dst_width, 3), dtype=np.float32)
    
    # 提取仿射变换矩阵参数
    m_x1, m_y1, m_z1 = d2s_matrix[0]
    m_x2, m_y2, m_z2 = d2s_matrix[1]
    
    const_value = 128
    
    for dy in range(dst_height):
        for dx in range(dst_width):
            # 通过仿射变换计算源图像中的对应坐标
            src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5
            src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5
            
            if src_x <= -1 or src_x >= src_width or src_y <= -1 or src_y >= src_height:
                # 源坐标超出图像范围，使用常数值填充
                c0 = c1 = c2 = const_value
            else:
                # 双线性插值
                y_low = int(np.floor(src_y))
                x_low = int(np.floor(src_x))
                y_high = y_low + 1
                x_high = x_low + 1
                
                ly = src_y - y_low
                lx = src_x - x_low
                hy = 1 - ly
                hx = 1 - lx
                
                w1 = hy * hx
                w2 = hy * lx
                w3 = ly * hx
                w4 = ly * lx
                
                # 获取四个相邻像素值
                v1 = v2 = v3 = v4 = [const_value, const_value, const_value]
                
                if y_low >= 0:
                    if x_low >= 0:
                        v1 = img[y_low, x_low]
                    if x_high < src_width:
                        v2 = img[y_low, x_high]
                
                if y_high < src_height:
                    if x_low >= 0:
                        v3 = img[y_high, x_low]
                    if x_high < src_width:
                        v4 = img[y_high, x_high]
                
                # 双线性插值计算
                c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]
                c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]
                c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]
            
            # BGR to RGB (交换B和R通道)
            c0, c2 = c2, c0
            
            # 归一化到[0,1]
            dst[dy, dx, 0] = c0 / 255.0
            dst[dy, dx, 1] = c1 / 255.0
            dst[dy, dx, 2] = c2 / 255.0
    
    return dst

def warpaffine_preprocess(img, input_size=(1280, 1280)):
    """使用与C++版本相同的仿射变换预处理"""
    kInputH, kInputW = input_size
    
    # 计算缩放比例（与preprocess.cu一致）
    scale = min(kInputH / img.shape[0], kInputW / img.shape[1])
    
    # 构建仿射变换矩阵（与preprocess.cu完全一致）
    s2d = np.array([
        [scale, 0, -scale * img.shape[1] * 0.5 + kInputW * 0.5],
        [0, scale, -scale * img.shape[0] * 0.5 + kInputH * 0.5]
    ], dtype=np.float32)
    
    # 计算逆变换矩阵
    d2s = cv2.invertAffineTransform(s2d)
    
    # 使用自定义的CUDA风格仿射变换
    preprocessed = cuda_like_warpaffine(img, d2s, kInputW, kInputH)
    
    # 转换为CHW格式
    preprocessed = preprocessed.transpose(2, 0, 1)  # HWC to CHW
    
    print(f"Python - Scale: {scale}, Input size: {kInputW}x{kInputH}, Image size: {img.shape[1]}x{img.shape[0]}")
    print(f"Python - s2d matrix: [{s2d[0,0]:.6f}, {s2d[0,1]:.6f}, {s2d[0,2]:.6f}]")
    print(f"Python -            [{s2d[1,0]:.6f}, {s2d[1,1]:.6f}, {s2d[1,2]:.6f}]")
    print(f"Python - d2s matrix: [{d2s[0,0]:.6f}, {d2s[0,1]:.6f}, {d2s[0,2]:.6f}]")
    print(f"Python -            [{d2s[1,0]:.6f}, {d2s[1,1]:.6f}, {d2s[1,2]:.6f}]")
    
    return preprocessed, d2s

def run_inference(engine, image):
    context = engine.create_execution_context()
    stream = cuda.Stream()
    # 注意：warpaffine_preprocess已经完成了BGR->RGB转换、归一化和通道转置
    # 这里不需要再次处理，直接使用预处理后的图像
    image = np.expand_dims(image, axis=0)

    input_idx = engine.get_binding_index("images")
    output_idx = engine.get_binding_index("output0")

    context.set_binding_shape(input_idx, image.shape)
    input_data_host = np.ascontiguousarray(image, dtype=np.float32)
    input_data_device = cuda.mem_alloc(input_data_host.nbytes)
    output_shape = tuple(context.get_binding_shape(output_idx))
    output_data_device = cuda.mem_alloc(int(np.prod(output_shape) * 4))
    output_data_host = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_htod_async(input_data_device, input_data_host, stream)
    bindings = [int(input_data_device), int(output_data_device)]
    context.execute_async_v2(bindings, stream.handle)
    cuda.memcpy_dtoh_async(output_data_host, output_data_device, stream)
    stream.synchronize()

    # 确保bin目录存在
    os.makedirs("bin", exist_ok=True)
    output_data_host.tofile("bin/trt_raw_output.bin")  # 直接保存为二进制文件
    print(f"✅ 推理结果已保存到 bin/trt_raw_output.bin")
    print(f"📌 输出 shape: {output_data_host.shape}, dtype: {output_data_host.dtype}")

    return output_data_host

def affine_transform_coords(coords, d2s_matrix, img0_shape):
    """使用仿射逆变换矩阵转换坐标（与C++版本一致）"""
    m = d2s_matrix.flatten()
    
    # 转换每个检测框的坐标
    for i in range(len(coords)):
        x1, y1, x2, y2 = coords[i][:4]
        
        # 应用仿射逆变换（与C++版本完全一致）
        new_x1 = m[0] * x1 + m[1] * y1 + m[2]
        new_y1 = m[3] * x1 + m[4] * y1 + m[5]
        new_x2 = m[0] * x2 + m[1] * y2 + m[2]
        new_y2 = m[3] * x2 + m[4] * y2 + m[5]
        
        print(f"Before transform: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")
        print(f"After transform: x1={new_x1:.2f}, y1={new_y1:.2f}, x2={new_x2:.2f}, y2={new_y2:.2f}")
        
        # 确保坐标在图像范围内
        new_x1 = max(0.0, min(img0_shape[1], new_x1))
        new_y1 = max(0.0, min(img0_shape[0], new_y1))
        new_x2 = max(0.0, min(img0_shape[1], new_x2))
        new_y2 = max(0.0, min(img0_shape[0], new_y2))
        
        # 确保x1 < x2 和 y1 < y2
        if new_x1 > new_x2:
            new_x1, new_x2 = new_x2, new_x1
        if new_y1 > new_y2:
            new_y1, new_y2 = new_y2, new_y1
            
        coords[i][:4] = [new_x1, new_y1, new_x2, new_y2]
    
    return coords

def decode_detections(pred, image_shape, img0_shape, d2s_matrix, conf_thresh=0.25):
    pred = np.squeeze(pred, axis=0)
    box_centers = pred[0:2, :]
    box_sizes = pred[2:4, :]
    confidences = pred[4, :]
    
    print(f"Total detections before filtering: {len(confidences)}")
    print(f"Confidence range: {confidences.min():.4f} - {confidences.max():.4f}")
    
    mask = confidences > conf_thresh
    if not np.any(mask):
        return np.empty((0, 6))
    
    box_centers = box_centers[:, mask]
    box_sizes = box_sizes[:, mask]
    confidences = confidences[mask]
    
    print(f"Detections after confidence filtering: {len(confidences)}")
    
    n = box_centers.shape[1]
    boxes = np.zeros((4, n))
    boxes[0, :] = box_centers[0, :] - box_sizes[0, :] / 2
    boxes[1, :] = box_centers[1, :] - box_sizes[1, :] / 2
    boxes[2, :] = box_centers[0, :] + box_sizes[0, :] / 2
    boxes[3, :] = box_centers[1, :] + box_sizes[1, :] / 2
    
    class_ids = np.zeros_like(confidences, dtype=np.int32)
    
    # 保存解码后、NMS前的结果（模拟C++格式）
    decode_before_nms = np.zeros(1 + len(confidences) * 7, dtype=np.float32)
    decode_before_nms[0] = len(confidences)  # 检测框数量
    for i in range(len(confidences)):
        base_idx = 1 + i * 7
        decode_before_nms[base_idx:base_idx+7] = [
            boxes[0, i], boxes[1, i], boxes[2, i], boxes[3, i],  # x1, y1, x2, y2
            confidences[i], class_ids[i], 1  # conf, class, keep_flag
        ]
    
    decode_before_nms.tofile("bin/python_decode_output.bin")
    print(f"✅ Python 解码输出已保存到 bin/python_decode_output.bin")
    print(f"📌 解码后检测框数量: {len(confidences)}")
    
    # 使用仿射变换转换坐标
    boxes = affine_transform_coords(boxes.T, d2s_matrix, img0_shape)
    boxes = np.array(boxes)
    
    # 过滤无效的检测框
    valid_mask = []
    for i, (x1, y1, x2, y2) in enumerate(boxes[:, :4]):
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            valid_mask.append(i)
        else:
            print(f"Filtering invalid box {i}: width={x2-x1:.2f}, height={y2-y1:.2f}")
    
    if len(valid_mask) == 0:
        return np.empty((0, 6))
    
    boxes = boxes[valid_mask]
    confidences = confidences[valid_mask]
    class_ids = class_ids[valid_mask]
    
    print(f"Valid detections after coordinate transform: {len(boxes)}")
    
    xywh_boxes = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in boxes[:, :4]]
    conf_list = confidences.tolist()
    indices = NMSBoxes(bboxes=xywh_boxes, scores=conf_list, score_threshold=conf_thresh, nms_threshold=0.5)
    
    if len(indices) == 0:
        return np.empty((0, 6))
    
    indices = np.array(indices).reshape(-1)
    selected_boxes = boxes[indices]
    selected_confs = confidences[indices]
    selected_classes = class_ids[indices]
    
    print(f"Final detections after NMS: {len(selected_boxes)}")
    
    # 保存NMS后的最终结果（模拟C++格式）
    nms_result = np.zeros(1 + len(selected_boxes) * 7, dtype=np.float32)
    nms_result[0] = len(selected_boxes)  # 检测框数量
    for i in range(len(selected_boxes)):
        base_idx = 1 + i * 7
        nms_result[base_idx:base_idx+7] = [
            selected_boxes[i, 0], selected_boxes[i, 1], selected_boxes[i, 2], selected_boxes[i, 3],  # x1, y1, x2, y2
            selected_confs[i], selected_classes[i], 1  # conf, class, keep_flag
        ]
    
    nms_result.tofile("bin/python_nms_output.bin")
    print(f"✅ Python NMS输出已保存到 bin/python_nms_output.bin")
    print(f"📌 NMS后检测框数量: {len(selected_boxes)}")
    
    # 打印前几个检测框的详细信息
    count = min(len(selected_boxes), 5)
    for i in range(count):
        box = selected_boxes[i]
        print(f"框 {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}], 置信度: {selected_confs[i]:.3f}, 类别: {int(selected_classes[i])}, keep: 1")
    
    return np.hstack((selected_boxes, selected_confs[:, None], selected_classes[:, None]))

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw(img, boxinfo, classes):
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    colors = [[160, 32 ,240],[255, 0, 0]]
    for detection in boxinfo:
        xyxy = detection[:4]
        conf = detection[4]
        cls = int(detection[5])
        if cls >= len(classes):
            continue
        label = f'{classes[cls]} {conf:.2f}'
        plot_one_box(x=xyxy, img=img, label=label, color=colors[cls], line_thickness=1)
    # cv2.imwrite("results.jpg", img)

def process_single_image(image_path, engine, classes, save_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[警告] 无法读取图片: {image_path}")
        return

    print(f"\n=== 处理图片: {os.path.basename(image_path)} ===")
    print(f"原始图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 使用与C++版本相同的仿射变换预处理
    preprocessed_image, d2s_matrix = warpaffine_preprocess(image, (1280, 1280))
    
    # 推理
    yolo_output = run_inference(engine, preprocessed_image)
    
    # 使用仿射逆变换解码检测结果
    results = decode_detections(yolo_output, image_shape=(1280, 1280), 
                               conf_thresh=0.25, img0_shape=image.shape, 
                               d2s_matrix=d2s_matrix)

    filename = os.path.basename(image_path)
    save_path = os.path.join(save_dir, filename)
    
    if len(results) > 0:
        print(f"检测到 {len(results)} 个目标")
        for i, detection in enumerate(results):
            x1, y1, x2, y2, conf, cls = detection
            print(f"目标 {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], 置信度: {conf:.3f}")
        
        draw(image, results, classes)
        cv2.imwrite(save_path, image)
        print(f"[成功] 检测完成: {filename}")
    else:
        cv2.imwrite(save_path, image)
        print(f"[提示] 未检测到目标: {filename}")
    
    print("=" * 50)

def main():
    # 清空并创建bin文件夹
    import shutil
    if os.path.exists("bin"):
        shutil.rmtree("bin")
    os.makedirs("bin", exist_ok=True)
    print("✅ 已清空并重新创建bin文件夹")
    
    # yolo_onnx = r"\\192.168.1.105\f\wangbh_project\ultralytics-main\runs\detect\roadsign_20250329\weights\best.onnx"
    # yolo_trt = r"\\192.168.1.105\f\wangbh_project\ultralytics-main\runs\detect\roadsign_20250329\weights\best.trt"
    # image_dir = r"D:\Yy\Pictures\20\biaopai"  # 输入图片文件夹
    # save_dir = r"D:\Yy\Pictures\20\biaopai_results"  # 输出保存文件夹

    # yolo_onnx = r"\\192.168.1.105\e\ecunits\Projects\yolov8\runs\detect\Pothole_20250626\weights\Pothole_20250626.onnx"
    # yolo_trt = r"\\192.168.1.105\e\ecunits\Projects\yolov8\runs\detect\Pothole_20250626\weights\Pothole_20250626.trt"
    # image_dir = r"D:\Yy\Documents\1.22\111111111111\222"  # 输入图片文件夹
    # save_dir = r"D:\Yy\Documents\1.22\111111111111\222results"  # 输出保存文件夹


    yolo_onnx = r"\\192.168.1.105\e\ecunits\Projects\yolov8\runs\detect\Abandoned_V7_20250625\weights\Abandoned_V7_20250625.onnx"
    yolo_trt = r"D:/Yy/Projects/Yolo_Infer/build/windows/x64/releasedbg/Abandoned_V7_20250625.trtmodel"

    # 使用当前项目中的测试图片
    test_image = r"D:/Yy/Projects/Yolo_Infer/build/windows/x64/releasedbg/2025042416192465632.jpg"
    save_dir = r"D:/Yy/Projects/Yolo_Infer/python_results"  # 输出文件夹路径


    os.makedirs(save_dir, exist_ok=True)

    classes = ['1', '2']

    if not os.path.exists(yolo_trt):
        print("[提示] TensorRT 引擎不存在，开始构建...")
        build_engine(yolo_onnx, yolo_trt)

    yolo_engine = load_engine(yolo_trt)

    # 检查测试图片是否存在
    if not os.path.exists(test_image):
        print(f"[错误] 测试图片不存在: {test_image}")
        return

    print(f"[开始] 使用Python版本测试图片: {test_image}")
    print("[注意] 使用与C++版本相同的仿射变换预处理")
    
    # 处理单张测试图片
    process_single_image(test_image, yolo_engine, classes, save_dir)

    print("[完成] Python版本测试完成！")
    print(f"[结果] 输出保存在: {save_dir}")

if __name__ == "__main__":
    main()






