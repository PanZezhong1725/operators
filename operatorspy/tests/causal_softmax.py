from ctypes import c_void_p
import ctypes
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    MutableTensor,
    DeviceEnum,
)

from operatorspy.tests.test_utils import get_args
import torch


def causal_softmax(x):
    type = x.dtype
    mask = torch.tril(torch.ones_like(x))
    y = x.clone()
    masked = torch.where(mask == 0, -torch.inf, y.to(torch.float32))
    return torch.nn.functional.softmax(masked, dim=-1).to(type)


def test(lib, descriptor, torch_device):
    x = torch.rand((5, 32, 1999), dtype=torch.float16).to(torch_device)

    ans = causal_softmax(x)
    lib.causalSoftmax(descriptor, to_tensor(x), None)

    assert torch.allclose(x, ans, atol=1, rtol=1e-3)
    print("Test passed!")


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    config = None
    descriptor = lib.createCausalSoftmaxDescriptor(device, config)
    test(lib, descriptor, "cpu")
    lib.destroyCausalSoftmaxDescriptor(descriptor)


def test_cuda(lib):
    device = DeviceEnum.DEVICE_CUDA

    class CausalSoftmaxCudaConfig(ctypes.Structure):
        _fields_ = [("max_dim", ctypes.c_uint)]

    config = ctypes.byref(CausalSoftmaxCudaConfig(max_dim=4096))
    descriptor = lib.createCausalSoftmaxDescriptor(device, config)
    test(lib, descriptor, "cuda")
    lib.destroyCausalSoftmaxDescriptor(descriptor)


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    lib.createCausalSoftmaxDescriptor.restype = c_void_p
    lib.destroyCausalSoftmaxDescriptor.argtypes = [c_void_p]
    lib.causalSoftmax.argtypes = [
        c_void_p,
        MutableTensor,
        c_void_p,
    ]
    if args.cpu:
        test_cpu(lib)
    if args.cuda:
        test_cuda(lib)
