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


def matmul(c, beta, a, b, alpha):
    input_dtype = c.dtype
    return (
        alpha * torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(input_dtype)
        + beta * c
    )


def test(lib, descriptor, torch_device):
    c = torch.zeros((1, 2048), dtype=torch.float16).to(torch_device)
    a = torch.rand((1, 2048), dtype=torch.float16).to(torch_device)
    b = torch.rand((2048, 2048), dtype=torch.float16).to(torch_device)

    beta = 0.0
    alpha = 1.0

    ans = matmul(c, beta, a, b, alpha)
    lib.matmul(
        descriptor,
        to_tensor(c, lib),
        beta,
        to_tensor(a, lib),
        to_tensor(b, lib),
        alpha,
        None,
    )

    assert torch.allclose(c, ans, atol=0, rtol=1e-3)
    print("Test passed!")


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    descriptor = lib.createMatmulDescriptor(device, None)
    test(lib, descriptor, "cpu")
    lib.destroyMatmulDescriptor(descriptor)


def test_cuda(lib):
    device = DeviceEnum.DEVICE_CUDA

    descriptor = lib.createMatmulDescriptor(device, None)
    test(lib, descriptor, "cuda")
    lib.destroyMatmulDescriptor(descriptor)

def test_bang(lib):
    import torch_mlu
    device = DeviceEnum.DEVICE_BANG
    descriptor = lib.createMatmulDescriptor(device, None)
    test(lib, descriptor, "mlu")
    lib.destroyMatmulDescriptor(descriptor)

if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    lib.createMatmulDescriptor.restype = c_void_p
    lib.destroyMatmulDescriptor.argtypes = [c_void_p]
    lib.matmul.argtypes = [
        c_void_p,
        CTensor,
        c_float,
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
