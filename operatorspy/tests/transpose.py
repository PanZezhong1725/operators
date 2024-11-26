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
    rearrange_tensor,
)

from operatorspy.tests.test_utils import get_args
import torch
from typing import Tuple

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = True
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class TransposeDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopTransposeDescriptor_t = POINTER(TransposeDescriptor)


# Output the permuted shape given the input shape and the perm order
def permute_shape(original_shape, perm=None):
    if perm is not None and len(original_shape) != len(perm):
        raise ValueError("The permutation must have the same length as the original shape.")
    
    if perm is None:
        # Infer the reverse perm
        perm = tuple(range(len(original_shape) - 1, -1, -1))
        permuted_shape = original_shape[::-1]
    else:
        permuted_shape = tuple(original_shape[i] for i in perm)
    
    return permuted_shape, perm


# convert a python tuple to a ctype void pointer
def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)


def transpose(x, perm):
    if PROFILE:
        ans = torch.permute(x, perm).clone()
        torch.cuda.synchronize()
        return ans
    return torch.permute(x, perm)


def test(
    lib,
    handle,
    torch_device,
    x_shape,
    perm,
    x_stride=None, 
    y_stride=None, 
    tensor_dtype=torch.float16,
):
    y_shape, perm = permute_shape(x_shape, perm)
    print(
        f"Testing Transpose on {torch_device} with x_shape:{x_shape} y_shape:{y_shape} perm:{perm} x_stride:{x_stride} y_stride:{y_stride} dtype:{tensor_dtype}"
    )

    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.rand(y_shape, dtype=tensor_dtype).to(torch_device)

    if x_stride is not None:
        x = rearrange_tensor(x, x_stride)
    if y_stride is not None:
        y = rearrange_tensor(y, y_stride)
    
    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = transpose(x, perm)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = transpose(x, perm)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    descriptor = infiniopTransposeDescriptor_t()

    check_error(
        lib.infiniopCreateTransposeDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            tuple_to_void_p(perm),
            len(perm),
        )
    )
    
    for i in range(NUM_PRERUN if PROFILE else 1):
        lib.infiniopTranspose(
            descriptor, y_tensor.data, x_tensor.data, None
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            lib.infiniopTranspose(
                descriptor, y_tensor.data, x_tensor.data, None
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")
    
    # print(" - x:\n", x, "\n - y:\n", y, "\n - ans:\n", ans)
    assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyTransposeDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for y_shape, x_shape, y_stride, x_stride in test_cases:
        test(lib, handle, "cpu", y_shape, x_shape, y_stride, x_stride, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", y_shape, x_shape, y_stride, x_stride, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for y_shape, x_shape, y_stride, x_stride in test_cases:
        test(lib, handle, "cuda", y_shape, x_shape, y_stride, x_stride, tensor_dtype=torch.float16)
        test(lib, handle, "cuda", y_shape, x_shape, y_stride, x_stride, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for y_shape, x_shape, y_stride, x_stride in test_cases:
        test(lib, handle, "mlu", y_shape, x_shape, y_stride, x_stride, tensor_dtype=torch.float16)
        test(lib, handle, "mlu", y_shape, x_shape, y_stride, x_stride, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, perm, x_stride, y_stride
        ((2, 2), (1, 0), None, None),
        ((1, 3), (1, 0), None, None),
        ((1, 13, 1, 1, 1), (1, 0, 4, 2, 3), None, None),
        ((5, 4, 3), (1, 0, 2), None, None),
        ((128, 648), (0, 1), None, None),
        ((128, 648), (1, 0), None, None),
        ((5, 4, 3), None, None, None),
        # ((32, 256, 112, 112), None, None, None),
        # ((32, 256, 112, 112), (2, 1, 3, 0), None, None),
        ((2048, 2048), None, (4096, 1), None),
        ((2048, 2048), None, None, (4096, 1)),
        ((2048, 2048), None, (4096, 1), (4096, 1)),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateTransposeDescriptor.restype = c_int32
    lib.infiniopCreateTransposeDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopTransposeDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_uint64,
    ]
    lib.infiniopTranspose.restype = c_int32
    lib.infiniopTranspose.argtypes = [
        infiniopTransposeDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyTransposeDescriptor.restype = c_int32
    lib.infiniopDestroyTransposeDescriptor.argtypes = [
        infiniopTransposeDescriptor_t,
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
