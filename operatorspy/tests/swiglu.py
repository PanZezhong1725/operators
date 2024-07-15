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

def swiglu(gate, up):
    return up * torch.nn.functional.silu(gate).to(gate.dtype)


def test(lib, descriptor, torch_device):
    gate = torch.rand((1, 64), dtype=torch.float16).to(torch_device)
    up = torch.rand((1, 64), dtype=torch.float16).to(torch_device)
    print(up)
    ans = swiglu(gate, up)
    lib.swiglu(descriptor, to_tensor(gate, lib), to_tensor(up, lib), None)
    print(ans)
    print(gate)
    assert torch.allclose(gate, ans, atol=1e-3, rtol=1e-3)
    print("Test passed!")


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    descriptor = lib.createSwigluDescriptor(device, None)
    test(lib, descriptor, "cpu")
    lib.destroySwigluDescriptor(descriptor)


def test_cuda(lib):
    device = DeviceEnum.DEVICE_CUDA

    descriptor = lib.createSwigluDescriptor(device, None)
    test(lib, descriptor, "cuda")
    lib.destroySwigluDescriptor(descriptor)


def test_bang(lib):
    import torch_mlu
    device = DeviceEnum.DEVICE_BANG
    descriptor = lib.createSwigluDescriptor(device, None)
    test(lib, descriptor, "mlu")
    lib.destroySwigluDescriptor(descriptor)


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    lib.createSwigluDescriptor.restype = c_void_p
    lib.destroySwigluDescriptor.argtypes = [c_void_p]
    lib.swiglu.argtypes = [
        c_void_p,
        CTensor,
        CTensor,
        c_void_p,
    ]
    if args.cpu:
        test_cpu(lib)
    if args.cuda:
        test_cuda(lib)
    if args.bang:
        test_bang(lib)
