from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64
import ctypes
import sys
import os

# 调整路径以导入 operatorspy 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
from enum import Enum, auto
import torch


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    # 对于 concat 算子，通常不支持 in-place 操作，因此这里只保留 OUT_OF_PLACE
    # 你可以根据实际需求扩展其他选项
    # INPLACE_A = auto()
    # INPLACE_B = auto()


class ConcatDescriptor(Structure):
    _fields_ = [("device", c_int32),]


infiniopConcatDescriptor_t = POINTER(ConcatDescriptor)


def concat_py(*tensors, dim=0):
    """使用 PyTorch 进行拼接的辅助函数"""
    return torch.cat(tensors, dim=dim)


def test(
    lib,
    handle,
    torch_device,
    c_shape,
    axis,
    input_shapes,
    tensor_dtype=torch.float32,
    inplace=Inplace.OUT_OF_PLACE,
):
    """
    测试 concat 算子
    """
    print(
        f"Testing Concat on {torch_device} with output_shape:{c_shape}, input_shapes:{input_shapes}, axis:{axis}, dtype:{tensor_dtype}, inplace: {inplace.name}"
    )
    
    # 创建输入张量
    inputs = [torch.rand(shape, dtype=tensor_dtype).to(torch_device) for shape in input_shapes]

    for idx, tensor in enumerate(inputs):
        print(f"Input {idx}:")
        print(tensor)
        print("-" * 50)  
    
    # 创建输出张量
    if inplace == Inplace.OUT_OF_PLACE:
        c = torch.zeros(c_shape, dtype=tensor_dtype).to(torch_device)
    else:
        # 对于 concat，通常不支持 in-place 操作，因此这里简化为 OUT_OF_PLACE
        c = torch.zeros(c_shape, dtype=tensor_dtype).to(torch_device)
    
    # 使用 PyTorch 进行拼接，作为参考答案
    ans = concat_py(*inputs, dim=axis)

    print("ans:",ans)
    print("-" * 50)  
    
    # 将张量转换为 infiniop 所需的格式
    input_tensors = [to_tensor(t, lib) for t in inputs]
    c_tensor = to_tensor(c, lib) if inplace == Inplace.OUT_OF_PLACE else to_tensor(c, lib)
    
    # 创建 Concat 描述符
    descriptor = infiniopConcatDescriptor_t()
    
    # 准备输入描述符数组
    num_inputs = len(input_tensors)
    input_desc_array_type = infiniopTensorDescriptor_t * num_inputs
    input_desc_array = input_desc_array_type(*[t.descriptor for t in input_tensors])
    
    # 创建描述符
    check_error(
        lib.infiniopCreateConcatDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,            # 使用 c_tensor 的描述符
            input_desc_array,               # 输入张量描述符数组
            c_uint64(num_inputs),
            c_uint64(axis),
        )
    )

    print("c1:",c)
    print("-" * 50)  

    # 执行拼接操作
    input_data_ptrs = (c_void_p * num_inputs)(*[t.data for t in input_tensors])
    check_error(
        lib.infiniopConcat(
            descriptor,
            c_tensor.data,
            ctypes.cast(input_data_ptrs, POINTER(c_void_p)),
            None  # 假设不需要流
        )
    )
    
    print("c2:",c)
    print("-" * 50)  

    # 验证结果
    assert torch.allclose(c, ans, atol=0, rtol=1e-5), "Concat result does not match PyTorch's result."
    
    # 销毁描述符
    check_error(lib.infiniopDestroyConcatDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for c_shape, axis, input_shapes, inplace in test_cases:
        test(lib, handle, "cpu", c_shape, axis, input_shapes, inplace=inplace)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for c_shape, axis, input_shapes, inplace in test_cases:
        test(lib, handle, "cuda", c_shape, axis, input_shapes, inplace=inplace)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu
    
    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for c_shape, axis, input_shapes, inplace in test_cases:
        test(lib, handle, "mlu", c_shape, axis, input_shapes, inplace=inplace)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    # 定义测试用例
    test_cases = [
        # (output_shape, axis, input_shapes, inplace)

        ((6, 3), 0, [(2, 3), (4, 3)], Inplace.OUT_OF_PLACE),
        # ((3, 6), 1, [(3, 2), (3, 4)], Inplace.OUT_OF_PLACE),
        # ((3, 7), 1, [(3, 2), (3, 4), (3,1)], Inplace.OUT_OF_PLACE),
        # ((3, 3, 10), 2, [(3, 3, 4), (3, 3, 6)], Inplace.OUT_OF_PLACE),
        # ((1, 1), 0, [(1, 1)], Inplace.OUT_OF_PLACE),
        # ((4, 5, 6), 0, [(1, 5, 6), (3, 5, 6)], Inplace.OUT_OF_PLACE),
        # ((2, 3, 6), 2, [(2, 3, 2), (2, 3, 4)], Inplace.OUT_OF_PLACE),

        # 添加更多测试用例以覆盖不同的维度和拼接轴
        # ((2, 10, 3), 1, [(2, 5, 3), (2, 2, 3),(2,3,3)], Inplace.OUT_OF_PLACE),  # 拼接沿第二维
    ]
    
    args = get_args()
    lib = open_lib()
    
    # 绑定 C++ 函数
    # 创建 Concat 描述符
    lib.infiniopCreateConcatDescriptor.restype = c_int32
    lib.infiniopCreateConcatDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopConcatDescriptor_t),
        infiniopTensorDescriptor_t,  # 输出张量描述符
        POINTER(infiniopTensorDescriptor_t),  # 输入张量描述符数组
        c_uint64,  # 输入张量数量
        c_uint64,  # 拼接轴
    ]
    
    # 执行 Concat
    lib.infiniopConcat.restype = c_int32
    lib.infiniopConcat.argtypes = [
        infiniopConcatDescriptor_t,
        c_void_p,  # 输出数据指针
        POINTER(c_void_p),  # 输入数据指针数组
        c_void_p,  # 流（假设为 NULL）
    ]
    
    # 销毁 Concat 描述符
    lib.infiniopDestroyConcatDescriptor.restype = c_int32
    lib.infiniopDestroyConcatDescriptor.argtypes = [
        infiniopConcatDescriptor_t,
    ]
    
    # 根据命令行参数执行测试
    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    
    print("\033[92mConcat Test passed!\033[0m")




