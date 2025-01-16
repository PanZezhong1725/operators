from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
import ctypes
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    CTensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
    rearrange_tensor,
)

from operatorspy.tests.test_utils import get_args, synchronize_device
from enum import Enum, auto
import torch

PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


class SwiGLUDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopSwiGLUDescriptor_t = POINTER(SwiGLUDescriptor)


def print_discrepancy(actual, expected, atol=0, rtol=1e-2):
    if actual.shape != expected.shape:
        raise ValueError("Tensors must have the same shape to compare.")

    # Calculate the difference mask based on atol and rtol
    diff_mask = torch.abs(actual - expected) > (atol + rtol * torch.abs(expected))
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)

    # Widths for columns (adjusted based on data)
    col_width = [18, 18, 18, 18]
    decimal_places = [0, 12, 12, 12]

    for idx in diff_indices:
        index_tuple = tuple(idx.tolist())
        print(
            f" > Index: {str(index_tuple):<{col_width[0]}}"
            f"actual: \033[31m{actual[index_tuple]:<{col_width[1]}.{decimal_places[1]}f}\033[0m"
            f"expect: \033[32m{expected[index_tuple]:<{col_width[2]}.{decimal_places[2]}f}\033[0m"
            f"delta: \033[33m{actual[index_tuple] - expected[index_tuple]:<{col_width[3]}.{decimal_places[3]}f}\033[0m"
        )

    return diff_indices


def swiglu(a, b):

    return a * b / (1 + torch.exp(-b.float()).to(b.dtype))


def test(
    lib,
    handle,
    torch_device,
    shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
    sync=None,
):
    print(
        f"Testing SwiGLU on {torch_device} with shape:{shape} a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} dtype:{dtype} "
        f"inplace:{inplace}"
    )
    a = torch.rand(shape, dtype=dtype).to(torch_device)
    b = torch.rand(shape, dtype=dtype).to(torch_device)
    c = (
        torch.rand(shape, dtype=dtype).to(torch_device)
        if inplace == Inplace.OUT_OF_PLACE
        else (a if inplace == Inplace.INPLACE_A else b)
    )

    if a_stride is not None:
        a = rearrange_tensor(a, a_stride)
    if b_stride is not None:
        b = rearrange_tensor(b, b_stride)
    if c_stride is not None and inplace == Inplace.OUT_OF_PLACE:
        c = rearrange_tensor(c, c_stride)

    c = (
        c
        if inplace == Inplace.OUT_OF_PLACE
        else (a if inplace == Inplace.INPLACE_A else b)
    )

    ans = swiglu(a, b)

    if sync is not None:
        sync()

    a_tensor = to_tensor(a, lib)
    b_tensor = to_tensor(b, lib)
    c_tensor = (
        to_tensor(c, lib)
        if inplace == Inplace.OUT_OF_PLACE
        else (a_tensor if inplace == Inplace.INPLACE_A else b_tensor)
    )
    descriptor = infiniopSwiGLUDescriptor_t()
    check_error(
        lib.infiniopCreateSwiGLUDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    a_tensor.descriptor.contents.invalidate()
    b_tensor.descriptor.contents.invalidate()
    c_tensor.descriptor.contents.invalidate()

    check_error(
        lib.infiniopSwiGLU(
            descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None
        )
    )

    assert torch.allclose(c, ans, atol=1e-4, rtol=1e-2)

    if PROFILE:
        # Profiling PyTorch implementation
        for i in range(NUM_PRERUN):
            _ = swiglu(a, b)
        synchronize_device(torch_device)
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = swiglu(a, b)
        synchronize_device(torch_device)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f" pytorch time: {elapsed * 1000 :6f} ms")

        # Profiling C Operators implementation
        for i in range(NUM_PRERUN):
            check_error(
                lib.infiniopSwiGLU(
                    descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None
                )
            )
        synchronize_device(torch_device)
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopSwiGLU(
                    descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None
                )
            )
        synchronize_device(torch_device)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"     lib time: {elapsed * 1000 :6f} ms")

    check_error(lib.infiniopDestroySwiGLUDescriptor(descriptor))


def test_cpu(lib, test_cases, tensor_dtypes):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)

    for shape, a_stride, b_stride, c_stride, inplace in test_cases:
        for tensor_dtype in tensor_dtypes:
            test(
                lib,
                handle,
                "cpu",
                shape,
                a_stride,
                b_stride,
                c_stride,
                tensor_dtype,
                inplace,
            )

    destroy_handle(lib, handle)


def test_cuda(lib, test_cases, tensor_dtypes):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)

    for shape, a_stride, b_stride, c_stride, inplace in test_cases:
        for tensor_dtype in tensor_dtypes:
            test(
                lib,
                handle,
                "cuda",
                shape,
                a_stride,
                b_stride,
                c_stride,
                tensor_dtype,
                inplace,
            )

    destroy_handle(lib, handle)


def test_bang(lib, test_cases, tensor_dtypes):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)

    for shape, a_stride, b_stride, c_stride, inplace in test_cases:
        for tensor_dtype in tensor_dtypes:
            test(
                lib,
                handle,
                "mlu",
                shape,
                a_stride,
                b_stride,
                c_stride,
                tensor_dtype,
                inplace,
            )

    destroy_handle(lib, handle)


def test_ascend(lib, test_cases, tensor_dtypes):
    import torch_npu

    device = DeviceEnum.DEVICE_ASCEND
    handle = create_handle(lib, device)

    for shape, a_stride, b_stride, c_stride, inplace in test_cases:
        for tensor_dtype in tensor_dtypes:
            test(
                lib,
                handle,
                "cpu",
                shape,
                a_stride,
                b_stride,
                c_stride,
                tensor_dtype,
                inplace,
                torch.npu.synchronize,
            )

    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # shape, a_stride, b_stride, c_stride, inplace
        ((13, 4), None, None, None, Inplace.OUT_OF_PLACE),
        ((13, 4), None, None, None, Inplace.INPLACE_A),
        ((13, 4), None, None, None, Inplace.INPLACE_B),
        ((13, 4), (10, 1), (10, 1), (10, 1), Inplace.OUT_OF_PLACE),
        ((16, 5632), None, None, None, Inplace.OUT_OF_PLACE),
        ((16, 5632), (13312, 1), (13312, 1), (13312, 1), Inplace.OUT_OF_PLACE),
        ((16, 5632), (13312, 1), (13312, 1), (13312, 1), Inplace.INPLACE_A),
        ((16, 5632), (13312, 1), (13312, 1), (13312, 1), Inplace.INPLACE_B),
    ]
    tensor_dtypes = [
        torch.float16,
    ]
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateSwiGLUDescriptor.restype = c_int32
    lib.infiniopCreateSwiGLUDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopSwiGLUDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopSwiGLU.restype = c_int32
    lib.infiniopSwiGLU.argtypes = [
        infiniopSwiGLUDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySwiGLUDescriptor.restype = c_int32
    lib.infiniopDestroySwiGLUDescriptor.argtypes = [
        infiniopSwiGLUDescriptor_t,
    ]

    if args.profile:
        PROFILE = True
    if args.cpu:
        test_cpu(lib, test_cases, tensor_dtypes)
    if args.cuda:
        test_cuda(lib, test_cases, tensor_dtypes)
    if args.bang:
        test_bang(lib, test_cases, tensor_dtypes)
    if args.ascend:
        test_ascend(lib, test_cases, tensor_dtypes)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases, tensor_dtypes)
    print("\033[92mTest passed!\033[0m")
