from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
import ctypes
import sys
import os

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

from operatorspy.tests.test_utils import get_args
import torch


class SwiGLUDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopSwiGLUDescriptor_t = POINTER(SwiGLUDescriptor)


def swiglu(a, b):
    return a * torch.nn.functional.silu(b.float()).to(b.dtype)


def test_out_of_place(
    lib,
    handle,
    torch_device,
    shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=torch.float16,
):
    print(
        f"Testing SwiGLU on {torch_device} with shape:{shape} a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} dtype:{dtype}"
    )
    a = torch.rand(shape, dtype=dtype).to(torch_device)
    b = torch.rand(shape, dtype=dtype).to(torch_device)
    c = torch.rand(shape, dtype=dtype).to(torch_device)
    ans = swiglu(a, b)

    if a_stride is not None:
        a = rearrange_tensor(a, a_stride)
    if b_stride is not None:
        b = rearrange_tensor(b, b_stride)
    if c_stride is not None:
        c = rearrange_tensor(c, c_stride)

    a_tensor = to_tensor(a, lib)
    b_tensor = to_tensor(b, lib)
    c_tensor = to_tensor(c, lib)
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
    lib.infiniopSwiGLU(descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None)

    assert torch.allclose(c, ans, atol=1e-3, rtol=1e-3)
    print("out-of-place Test passed!")

    check_error(lib.infiniopDestroySwiGLUDescriptor(descriptor))


def test_in_place1(
    lib,
    handle,
    torch_device,
    shape,
    a_stride=None,
    b_stride=None,
    dtype=torch.float16,
):
    a = torch.rand(shape, dtype=dtype).to(torch_device)
    b = torch.rand(shape, dtype=dtype).to(torch_device)
    ans = swiglu(a, b)

    if a_stride is not None:
        a = rearrange_tensor(a, a_stride)
    if b_stride is not None:
        b = rearrange_tensor(b, b_stride)

    a_tensor = to_tensor(a, lib)
    b_tensor = to_tensor(b, lib)
    descriptor = infiniopSwiGLUDescriptor_t()
    check_error(
        lib.infiniopCreateSwiGLUDescriptor(
            handle,
            ctypes.byref(descriptor),
            a_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )
    lib.infiniopSwiGLU(descriptor, a_tensor.data, a_tensor.data, b_tensor.data, None)

    assert torch.allclose(a, ans, atol=1e-3, rtol=1e-3)
    print("in-place1 Test passed!")

    check_error(lib.infiniopDestroySwiGLUDescriptor(descriptor))


def test_in_place2(
    lib,
    handle,
    torch_device,
    shape,
    a_stride=None,
    b_stride=None,
    dtype=torch.float16,
):
    a = torch.rand(shape, dtype=dtype).to(torch_device)
    b = torch.rand(shape, dtype=dtype).to(torch_device)
    ans = swiglu(a, b)

    if a_stride is not None:
        a = rearrange_tensor(a, a_stride)
    if b_stride is not None:
        b = rearrange_tensor(b, b_stride)

    a_tensor = to_tensor(a, lib)
    b_tensor = to_tensor(b, lib)
    descriptor = infiniopSwiGLUDescriptor_t()
    check_error(
        lib.infiniopCreateSwiGLUDescriptor(
            handle,
            ctypes.byref(descriptor),
            b_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )
    lib.infiniopSwiGLU(descriptor, b_tensor.data, a_tensor.data, b_tensor.data, None)

    assert torch.allclose(b, ans, atol=1e-3, rtol=1e-3)
    print("in-place2 Test passed!")

    check_error(lib.infiniopDestroySwiGLUDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)

    for shape, a_stride, b_stride, c_stride, dtype in test_cases:
        test_out_of_place(
            lib, handle, "cpu", shape, a_stride, b_stride, c_stride, dtype
        )
        test_in_place1(lib, handle, "cpu", shape, a_stride, b_stride, dtype)
        test_in_place2(lib, handle, "cpu", shape, a_stride, b_stride, dtype)

    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)

    for shape, a_stride, b_stride, c_stride, dtype in test_cases:
        test_out_of_place(
            lib, handle, "cuda", shape, a_stride, b_stride, c_stride, dtype
        )
        test_in_place1(lib, handle, "cuda", shape, a_stride, b_stride, dtype)
        test_in_place2(lib, handle, "cuda", shape, a_stride, b_stride, dtype)

    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu
    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)

    for shape, a_stride, b_stride, c_stride, dtype in test_cases:
        test_out_of_place(
            lib, handle, "mlu", shape, a_stride, b_stride, c_stride, dtype
        )
        test_in_place1(lib, handle, "mlu", shape, a_stride, b_stride, dtype)
        test_in_place2(lib, handle, "mlu", shape, a_stride, b_stride, dtype)

    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # shape, a_stride, b_stride, c_stride, dtype
        ((13, 4), None, None, None, torch.float16),
        ((13, 4), (10, 1), (10, 1), (10, 1), torch.float16),
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

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if args.ascend:
        test_ascend(lib)
