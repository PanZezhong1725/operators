from ctypes import c_float, c_void_p
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    CTensor,
    DeviceEnum,
)

from operatorspy.tests.test_utils import get_args
import torch


def rms_norm(x, w, eps):
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return w * hidden_states.to(input_dtype)


def test(lib, descriptor, torch_device):

    y = torch.zeros((5, 16), dtype=torch.float16).to(torch_device)
    x = torch.rand((5, 16), dtype=torch.float16).to(torch_device)
    w = torch.ones((16,), dtype=torch.float16).to(torch_device)

    eps = 1e-5
    ans = rms_norm(x, w, eps)
    lib.rmsNorm(
        descriptor, to_tensor(y), to_tensor(x), to_tensor(w), eps, None
    )

    assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    print("Test passed!")


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    descriptor = lib.createRMSNormDescriptor(device, None)
    test(lib, descriptor, "cpu")
    lib.destroyRMSNormDescriptor(descriptor)


def test_cuda(lib):
    device = DeviceEnum.DEVICE_CUDA
    descriptor = lib.createRMSNormDescriptor(device, None)
    test(lib, descriptor, "cuda")
    lib.destroyRMSNormDescriptor(descriptor)

def test_cnnl(lib):
    import torch_mlu
    device = DeviceEnum.DEVICE_MLU
    descriptor = lib.createRMSNormDescriptor(device, None)
    test(lib, descriptor, "mlu")
    lib.destroyRMSNormDescriptor(descriptor)


def test_npu(lib):
    device = DeviceEnum.DEVICE_NPU
    descriptor = lib.createRMSNormDescriptor(device, None)
    test(lib, descriptor, "cpu")
    lib.destroyRMSNormDescriptor(descriptor)


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    lib.createRMSNormDescriptor.restype = c_void_p
    lib.destroyRMSNormDescriptor.argtypes = [c_void_p]
    lib.rmsNorm.argtypes = [
        c_void_p,
        CTensor,
        CTensor,
        CTensor,
        c_float,
        c_void_p,
    ]
    if args.cpu:
        test_cpu(lib)
    if args.cuda:
        test_cuda(lib)
    if args.cnnl:
        test_cnnl(lib)
    if args.ascend:
        # import torch_npu
        # torch_npu.npu.set_device(0)
        test_npu(lib)
