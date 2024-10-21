from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p, c_float
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
    create_workspace,
)

from operatorspy.tests.test_utils import get_args
import torch


class MatmulDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopMatmulDescriptor_t = POINTER(MatmulDescriptor)

def matmul(c, beta, a, b, alpha):
    input_dtype = c.dtype
    return (
        alpha * torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(input_dtype)
        + beta * c
    )


def test(
    lib,
    handle,
    torch_device,
    alpha,
    beta,
    a_shape,
    b_shape,
    c_shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=torch.float16,
):
    print(
        f"Testing Matmul on {torch_device} with a_shape:{a_shape} b_shape:{b_shape} c_shape:{c_shape}"
        f" a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} dtype:{dtype}"
    )

    a = torch.rand(a_shape, dtype=dtype).to(torch_device)
    b = torch.rand(b_shape, dtype=dtype).to(torch_device)
    c = torch.zeros(c_shape, dtype=dtype).to(torch_device)

    if a_stride is not None:
        a = rearrange_tensor(a, a_stride)
    if b_stride is not None:
        b = rearrange_tensor(b, b_stride)
    if c_stride is not None:
        c = rearrange_tensor(c, c_stride)

    ans = matmul(c, beta, a, b, alpha)
    
    a_tensor = to_tensor(a, lib)
    b_tensor = to_tensor(b, lib)
    c_tensor = to_tensor(c, lib)
    descriptor = infiniopMatmulDescriptor_t()
    check_error(
        lib.infiniopCreateMatmulDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,
            alpha,
            a_tensor.descriptor,
            b_tensor.descriptor,
            beta
        )
    )

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetMatmulWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, a.device)

    check_error(
        lib.infiniopMatmul(
            descriptor,
            workspace.data_ptr() if workspace is not None else None,
            workspace_size.value,
            c_tensor.data,
            a_tensor.data,
            b_tensor.data,
            None,
        )
    )

    assert torch.allclose(c, ans, atol=0, rtol=1e-2)

    check_error(lib.infiniopDestroyMatmulDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)

    for (
        alpha,
        beta,
        a_shape,
        b_shape,
        c_shape,
        a_stride,
        b_stride,
        c_stride,
        dtype,
    ) in test_cases:
        test(
            lib,
            handle,
            "cpu",
            alpha,
            beta,
            a_shape,
            b_shape,
            c_shape,
            a_stride,
            b_stride,
            c_stride,
            dtype,
        )

    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)

    for (
        alpha,
        beta,
        a_shape,
        b_shape,
        c_shape,
        a_stride,
        b_stride,
        c_stride,
        dtype,
    ) in test_cases:
        test(
            lib,
            handle,
            "cuda",
            alpha,
            beta,
            a_shape,
            b_shape,
            c_shape,
            a_stride,
            b_stride,
            c_stride,
            dtype,
        )

    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu
    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)

    for (
        alpha,
        beta,
        a_shape,
        b_shape,
        c_shape,
        a_stride,
        b_stride,
        c_stride,
        dtype,
    ) in test_cases:
        test(
            lib,
            handle,
            "mlu",
            alpha,
            beta,
            a_shape,
            b_shape,
            c_shape,
            a_stride,
            b_stride,
            c_stride,
            dtype,
        )

    destroy_handle(lib, handle)

def test_ascend(lib, test_cases):
    import torch_npu

    device = DeviceEnum.DEVICE_ASCEND
    handle = create_handle(lib, device)

    for (
        alpha,
        beta,
        a_shape,
        b_shape,
        c_shape,
        a_stride,
        b_stride,
        c_stride,
        dtype,
    ) in test_cases:
        test(
            lib,
            handle,
            "npu",
            alpha,
            beta,
            a_shape,
            b_shape,
            c_shape,
            a_stride,
            b_stride,
            c_stride,
            dtype,
        )

    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # alpha, beta, a_shape, b_shape, c_shape, a_stride, b_stride, c_stride, dtype
        (1.0, 0.0, (1, 2048), (2048, 2048), (1, 2048), None, None, None, torch.float16),
        (
            1.0,
            0.0,
            (1, 2048),
            (2048, 2048),
            (1, 2048),
            (4096, 1),
            (4096, 1),
            (4096, 1),
            torch.float16,
        ),
    ]
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateMatmulDescriptor.restype = c_int32
    lib.infiniopCreateMatmulDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopMatmulDescriptor_t),
        infiniopTensorDescriptor_t,
        c_float,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float
    ]

    lib.infiniopGetMatmulWorkspaceSize.restype = c_int32
    lib.infiniopGetMatmulWorkspaceSize.argtypes = [
        infiniopMatmulDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopMatmul.restype = c_int32
    lib.infiniopMatmul.argtypes = [
        infiniopMatmulDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyMatmulDescriptor.restype = c_int32
    lib.infiniopDestroyMatmulDescriptor.argtypes = [
        infiniopMatmulDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if args.ascend:
        test_ascend(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang or args.ascend):
        test_cpu(lib, test_cases)
    print("Test passed!")
