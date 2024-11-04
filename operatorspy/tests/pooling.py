from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64
import ctypes
import sys
import os
import time

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
from typing import Tuple


class PoolingDescriptor(Structure):
    _fields_ = [("device", c_int32)]


class PoolingMode(Enum):
    MAX_POOL = 0
    AVG_POOL = 1


infiniopPoolingDescriptor_t = POINTER(PoolingDescriptor)


def pool(x, k, padding, stride, pooling_mode, dilation = 1):
    pooling_layers = {
        1: (torch.nn.MaxPool1d, torch.nn.AvgPool1d),
        2: (torch.nn.MaxPool2d, torch.nn.AvgPool2d),
        3: (torch.nn.MaxPool3d, torch.nn.AvgPool3d),
    }

    ndim = len(x.shape) - 2
    if ndim not in pooling_layers:
        print("Error: Pytorch -> Unsupported tensor dimension")
        return None

    max_pool, avg_pool = pooling_layers[ndim]
    if pooling_mode == PoolingMode.MAX_POOL:
        return max_pool(k, stride=stride, padding=padding, dilation=dilation)(x)
    else:
        return avg_pool(k, stride=stride, padding=padding)(x)


def inferShape(x_shape, kernel_shape, padding, strides):
    assert (
        len(x_shape) - 2 == len(kernel_shape) == len(padding) == len(strides)
    ), "kernel, pads, and strides should have the same length; the length of input x should be 2 more than that of kernel"
    input_shape = x_shape[2:]
    output_shape = []

    for dim, k, p, s in zip(input_shape, kernel_shape, padding, strides):
        output_dim = (dim + 2 * p - k) // s + 1
        output_shape.append(output_dim)

    return x_shape[:2] + tuple(output_shape)

# convert a python tuple to a ctype void pointer
def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)

def test(
    lib,
    handle,
    torch_device,
    x_shape, 
    k_shape, 
    padding,
    strides,
    tensor_dtype=torch.float16,
    pooling_mode=PoolingMode.MAX_POOL
):
    print(
        f"Testing Pooling on {torch_device} with x_shape:{x_shape} kernel_shape:{k_shape} padding:{padding} strides:{strides} dtype:{tensor_dtype} pooling_mode: {pooling_mode.name}"
    )

    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.rand(inferShape(x_shape, k_shape, padding, strides), dtype=tensor_dtype).to(torch_device)
    
    ans = pool(x, k_shape, padding, strides, pooling_mode)

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    descriptor = infiniopPoolingDescriptor_t()

    check_error(
        lib.infiniopCreatePoolingDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            tuple_to_void_p(k_shape),
            tuple_to_void_p(padding),
            tuple_to_void_p(strides),
            len(k_shape),
            pooling_mode.value,
        )
    )
    lib.infiniopPooling(
        descriptor, y_tensor.data, x_tensor.data, None
    )

    print(" - x :\n", x, "\n - y :\n", y, "\n - ans:\n", ans)
    assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyPoolingDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, kernel_shape, padding, strides, pooling_mode in test_cases:
        test(lib, handle, "cpu", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float16, pooling_mode=pooling_mode)
        test(lib, handle, "cpu", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float32, pooling_mode=pooling_mode)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape, kernel_shape, padding, strides, pooling_mode in test_cases:
        test(lib, handle, "cuda", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float16, pooling_mode=pooling_mode)
        test(lib, handle, "cuda", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float32, pooling_mode=pooling_mode)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for x_shape, kernel_shape, padding, strides, pooling_mode in test_cases:
        test(lib, handle, "mlu", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float16, pooling_mode=pooling_mode)
        test(lib, handle, "mlu", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float32, pooling_mode=pooling_mode)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, kernel_shape, padding, strides, pooling_mode
        # ((), (), (), (), PoolingMode.MAX_POOL),
        # ((1, 1, 10), (3,), (1,), (1,), PoolingMode.MAX_POOL),
        # ((1, 1, 10), (3,), (1,), (1,), PoolingMode.AVG_POOL),
        # ((1, 3, 224, 224), (3, 3), (1, 1), (2, 2), PoolingMode.MAX_POOL),
        # ((1, 3, 224, 224), (3, 3), (1, 1), (2, 2), PoolingMode.AVG_POOL),
        ((1, 1, 3, 3, 3), (5, 5, 5), (2, 2, 2), (2, 2, 2), PoolingMode.MAX_POOL),
        ((32, 3, 10, 10, 10), (5, 5, 5), (2, 2, 2), (2, 2, 2), PoolingMode.AVG_POOL),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreatePoolingDescriptor.restype = c_int32
    lib.infiniopCreatePoolingDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopPoolingDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint64,
        c_int32,
    ]
    lib.infiniopPooling.restype = c_int32
    lib.infiniopPooling.argtypes = [
        infiniopPoolingDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyPoolingDescriptor.restype = c_int32
    lib.infiniopDestroyPoolingDescriptor.argtypes = [
        infiniopPoolingDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
