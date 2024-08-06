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
    y = torch.zeros((16, 13312), dtype=torch.float16).to(torch_device)
    x = torch.rand((16, 2048), dtype=torch.float16).to(torch_device)
    w = torch.ones((2048,), dtype=torch.float16).to(torch_device)

    eps = 1e-5
    ans = rms_norm(x, w, eps)
    lib.rmsNorm(
        descriptor, to_tensor(y, lib, [16, 2048], [26624, 2]), to_tensor(x, lib), to_tensor(w, lib), eps, None
    )

    # print(ans)
    # print("=======================================================")
    # print(y[:, :2048])
    assert torch.allclose(y[:, :2048], ans, atol=1e-1, rtol=1e-3)
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

def test_bang(lib):
    import torch_mlu
    device = DeviceEnum.DEVICE_BANG
    descriptor = lib.createRMSNormDescriptor(device, None)
    test(lib, descriptor, "mlu")
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
    if args.bang:
        test_bang(lib)
