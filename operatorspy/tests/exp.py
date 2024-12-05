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
)

from operatorspy.tests.test_utils import get_args
from enum import Enum, auto
import torch

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


class ExpDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopExpDescriptor_t = POINTER(ExpDescriptor)


def exp(x):
    if PROFILE:
        ans = torch.exp(x).to(x.dtype)
        torch.cuda.synchronize()
        return ans
    return torch.exp(x).to(x.dtype)


def test(
    lib,
    handle,
    torch_device,
    tensor_shape, 
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Exp on {torch_device} with tensor_shape:{tensor_shape} dtype:{tensor_dtype} inplace: {inplace.name}"
    )

    x = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device) * 4 - 2
    y = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device) if inplace == Inplace.OUT_OF_PLACE else x

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = exp(x)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = exp(x)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS * 1000
        print(f"pytorch time: {elapsed :6f} ms")

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib) if inplace == Inplace.OUT_OF_PLACE else x_tensor
    descriptor = infiniopExpDescriptor_t()

    check_error(
        lib.infiniopCreateExpDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
        )
    )
    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(lib.infiniopExp(descriptor, y_tensor.data, x_tensor.data, None))
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopExp(descriptor, y_tensor.data, x_tensor.data, None)
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS * 1000
        print(f"    lib time: {elapsed :6f} ms")

    assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyExpDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for tensor_shape, inplace in test_cases:
        test(lib, handle, "cpu", tensor_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cpu", tensor_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for tensor_shape, inplace in test_cases:
        test(lib, handle, "cuda", tensor_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cuda", tensor_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for tensor_shape, inplace in test_cases:
        test(lib, handle, "mlu", tensor_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "mlu", tensor_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # tensor_shape, inplace
        ((), Inplace.OUT_OF_PLACE),
        ((), Inplace.INPLACE_X),
        ((1, 3), Inplace.OUT_OF_PLACE),
        ((3, 3), Inplace.OUT_OF_PLACE),
        ((3, 3, 13, 9, 17), Inplace.INPLACE_X),
        ((32, 15, 512), Inplace.OUT_OF_PLACE),
        ((33, 333, 333), Inplace.OUT_OF_PLACE),
        # ((32, 256, 112, 112), Inplace.OUT_OF_PLACE),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateExpDescriptor.restype = c_int32
    lib.infiniopCreateExpDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopExpDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopExp.restype = c_int32
    lib.infiniopExp.argtypes = [
        infiniopExpDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyExpDescriptor.restype = c_int32
    lib.infiniopDestroyExpDescriptor.argtypes = [
        infiniopExpDescriptor_t,
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
