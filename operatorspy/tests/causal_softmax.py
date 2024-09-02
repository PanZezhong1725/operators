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


class CausalSoftmaxDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopCausalSoftmaxDescriptor_t = POINTER(CausalSoftmaxDescriptor)


def causal_softmax(x):
    device = x.device
    type = x.dtype
    # x = x.cpu()
    mask = torch.tril(torch.ones_like(x), diagonal=-1).flip(dims=[-2, -1])
    y = x.clone()
    masked = torch.where(mask == 1.0, torch.tensor(-torch.inf).to(torch.float32).to(device), y.to(torch.float32))
    # return torch.nn.functional.softmax(masked, dim=-1).to(type)
    return torch.nn.functional.softmax(masked, dim=-1).to(type)


def test(lib, handle, torch_device, x_shape, x_stride=None, x_dtype=torch.float16):
    print(
        f"Testing CausalSoftmax on {torch_device} with x_shape:{x_shape} x_stride:{x_stride} dtype:{x_dtype}"
    )
    x = torch.rand(x_shape, dtype=x_dtype).to(torch_device)
    ans = causal_softmax(x)
    if x_stride is not None:
        x = rearrange_tensor(x, x_stride)
    x_tensor = to_tensor(x, lib)
    descriptor = infiniopCausalSoftmaxDescriptor_t()
    check_error(
        lib.infiniopCreateCausalSoftmaxDescriptor(
            handle, ctypes.byref(descriptor), x_tensor.descriptor
        )
    )
    workspace_size = ctypes.c_ulong(0)
    lib.infiniopGetCausalSoftmaxWorkspaceSize(descriptor, 
                                              ctypes.byref(workspace_size))
    workspace = to_tensor(torch.zeros(workspace_size.value, dtype=torch.int8).to(torch_device), lib)
    
    lib.infiniopCausalSoftmax(descriptor, workspace.data, workspace_size, x_tensor.data, None)
    print(ans[0])
    print(x[0])
    assert torch.allclose(x, ans, atol=0, rtol=1e-2)
    
    print("Test passed!")
    check_error(lib.infiniopDestroyCausalSoftmaxDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, x_stride in test_cases:
        test(lib, handle, "cpu", x_shape, x_stride)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape, x_stride in test_cases:
        test(lib, handle, "cuda", x_shape, x_stride)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for x_shape, x_stride in test_cases:
        test(lib, handle, "mlu", x_shape, x_stride)
    destroy_handle(lib, handle)
    
def test_ascend(lib, test_cases):
    import torch_npu

    device = DeviceEnum.DEVICE_NPU
    handle = create_handle(lib, device)
    for x_shape, x_stride in test_cases:
        test(lib, handle, "npu", x_shape, x_stride)
        
    # test(lib, handle, "npu")
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, x_stride
        ((32, 20, 512), None),
        ((32, 20, 512), (20480, 512, 1)),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateCausalSoftmaxDescriptor.restype = c_int32
    lib.infiniopCreateCausalSoftmaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopCausalSoftmaxDescriptor_t),
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetCausalSoftmaxWorkspaceSize.restype = c_int32
    lib.infiniopGetCausalSoftmaxWorkspaceSize.argtypes = [
        infiniopCausalSoftmaxDescriptor_t,
        POINTER(c_uint64),
    ]
    lib.infiniopCausalSoftmax.restype = c_int32
    lib.infiniopCausalSoftmax.argtypes = [
        infiniopCausalSoftmaxDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyCausalSoftmaxDescriptor.restype = c_int32
    lib.infiniopDestroyCausalSoftmaxDescriptor.argtypes = [
        infiniopCausalSoftmaxDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if args.ascend:
        test_ascend(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    print("Test passed!")
