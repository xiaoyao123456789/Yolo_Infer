import onnx
from onnx import helper, TensorProto, numpy_helper

def add_transpose_to_output0(onnx_path, new_onnx_path):
    # 加载原始模型
    model = onnx.load(onnx_path)
    graph = model.graph

    # 找到原始 output0 节点的输出名称
    original_output_name = "output0"
    new_transpose_output_name = "output0_transposed"

    # 添加 Transpose 节点（对 1,37,33600 进行 [0,2,1] 转置，得到 1,33600,37）
    transpose_node = helper.make_node(
        'Transpose',
        inputs=[original_output_name],
        outputs=[new_transpose_output_name],
        perm=[0, 2, 1],
        name='Transpose_output0'
    )
    graph.node.append(transpose_node)

    # 更新 graph 输出为新的 transpose 节点输出
    for i, output in enumerate(graph.output):
        if output.name == original_output_name:
            graph.output[i].name = new_transpose_output_name
            graph.output[i].type.tensor_type.shape.dim[1].dim_value = 33600
            graph.output[i].type.tensor_type.shape.dim[2].dim_value = 37

    # 保存修改后的模型
    onnx.save(model, new_onnx_path)
    print(f"✅ 成功保存新模型到: {new_onnx_path}")

# 使用示例
add_transpose_to_output0(r"D:\Yy\Projects\Yolo_Infer\build\windows\x64\releasedbg\Line_V2.onnx", r"D:\Yy\Projects\Yolo_Infer\build\windows\x64\releasedbg\Line_V2_Tra.onnx")
