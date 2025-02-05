from ctypes import POINTER, Structure, c_int32, c_void_p
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
    rearrange_tensor,
)

from operatorspy.tests.test_utils import get_args, synchronize_device
import torch

PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class ExpandDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopExpandDescriptor_t = POINTER(ExpandDescriptor)


def expand(x, y):
    return x.expand_as(y).clone()


def test(
    lib,
    handle,
    torch_device,
    y_shape, 
    x_shape,
    y_stride=None, 
    x_stride=None, 
    tensor_dtype=torch.float16,
):
    print(
        f"Testing Expand on {torch_device} with x_shape:{x_shape} y_shape:{y_shape} x_stride:{x_stride} y_stride:{y_stride} dtype:{tensor_dtype}"
    )

    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.rand(y_shape, dtype=tensor_dtype).to(torch_device)

    if x_stride is not None:
        x = rearrange_tensor(x, x_stride)
    if y_stride is not None:
        y = rearrange_tensor(y, y_stride)

    ans = expand(x, y)

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    descriptor = infiniopExpandDescriptor_t()

    check_error(
        lib.infiniopCreateExpandDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    check_error(lib.infiniopExpand(descriptor, y_tensor.data, x_tensor.data, None))

    assert torch.allclose(y, ans, atol=0, rtol=0)

    if PROFILE:
        # Profiling PyTorch implementation
        for i in range(NUM_PRERUN):
            _ = expand(x, y)
        synchronize_device(torch_device)
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = expand(x, y)
        synchronize_device(torch_device)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f" pytorch time: {elapsed * 1000 :6f} ms")

        # Profiling C Operators implementation
        for i in range(NUM_PRERUN):
            check_error(lib.infiniopExpand(descriptor, y_tensor.data, x_tensor.data, None))
        synchronize_device(torch_device)
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(lib.infiniopExpand(descriptor, y_tensor.data, x_tensor.data, None))
        synchronize_device(torch_device)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"     lib time: {elapsed * 1000 :6f} ms")

    check_error(lib.infiniopDestroyExpandDescriptor(descriptor))


def test_cpu(lib, test_cases, tensor_dtypes):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for y_shape, x_shape, y_stride, x_stride in test_cases:
        for tensor_dtype in tensor_dtypes:
            test(lib, handle, "cpu", y_shape, x_shape, y_stride, x_stride, tensor_dtype=tensor_dtype)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases, tensor_dtypes):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for y_shape, x_shape, y_stride, x_stride in test_cases:
        for tensor_dtype in tensor_dtypes:
            test(lib, handle, "cuda", y_shape, x_shape, y_stride, x_stride, tensor_dtype=tensor_dtype)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases, tensor_dtypes):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for y_shape, x_shape, y_stride, x_stride in test_cases:
        for tensor_dtype in tensor_dtypes:
            test(lib, handle, "mlu", y_shape, x_shape, y_stride, x_stride, tensor_dtype=tensor_dtype)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # y_shape, x_shape, y_stride, x_stride
        ((), (), None, None),
        ((3, 3), (1,), None, None),
        ((5, 4, 3), (4, 3,), None, (6, 1)),
        ((99, 111), (111,), None, None),
        ((2, 4, 3), (1, 3), None, None),
        ((2, 20, 3), (2, 1, 3), None, None),
        ((2, 3, 4, 5), (5,), None, None),
        ((3, 2, 4, 5), (3, 2, 1, 1), None, None),
        ((32, 256, 112, 112), (32, 256, 112, 1), None, None),
    ]
    tensor_dtypes = [
        torch.float16, torch.float32,
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateExpandDescriptor.restype = c_int32
    lib.infiniopCreateExpandDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopExpandDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopExpand.restype = c_int32
    lib.infiniopExpand.argtypes = [
        infiniopExpandDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyExpandDescriptor.restype = c_int32
    lib.infiniopDestroyExpandDescriptor.argtypes = [
        infiniopExpandDescriptor_t,
    ]

    if args.profile:
        PROFILE = True
    if args.cpu:
        test_cpu(lib, test_cases, tensor_dtypes)
    if args.cuda:
        test_cuda(lib, test_cases, tensor_dtypes)
    if args.bang:
        test_bang(lib, test_cases, tensor_dtypes)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases, tensor_dtypes)
    print("\033[92mTest passed!\033[0m")
