import ctypes
from ctypes import c_float, POINTER, c_void_p
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
import time


def test(lib, descriptor, torch_device, x = None):
    if x is None:
        x = torch.rand((10, 10), dtype=torch.float16).to(torch_device)
    else:
        x = x.to(torch_device)
    y = torch.zeros((5, 5), dtype=torch.float16).to(torch_device)

    lib.reform(descriptor, to_tensor(y, lib), to_tensor(x, lib, [5, 5], [20, 2]), None)
    
    return x, y

def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    config = None
    descriptor = lib.createReformDescriptor(device, config)
    test(lib, descriptor, "cpu")
    lib.destroyReformDescriptor(descriptor)
    print("Test passed!")

def run_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    config = None
    descriptor = lib.createReformDescriptor(device, config)
    x, ans = test(lib, descriptor, "cpu")
    lib.destroyReformDescriptor(descriptor)
    return x, ans

def test_cuda(lib):
    device = DeviceEnum.DEVICE_CUDA
    config = None
    descriptor = lib.createReformDescriptor(device, config)
    
    # compare with cpu results
    x, cpu_ans = run_cpu(lib)
    _, cuda_ans = test(lib, descriptor, "cuda", x)
    
    assert torch.allclose(cuda_ans.cpu(), cpu_ans, atol=1e-3, rtol=1e-3)
    print("Test passed!")

    lib.destroyReformDescriptor(descriptor)

def test_bang(lib):
    import torch_mlu
    device = DeviceEnum.DEVICE_BANG
    descriptor = lib.createReformDescriptor(device, None)
    
    # compare with cpu results
    x, cpu_ans = run_cpu(lib)
    _, bang_ans = test(lib, descriptor, "mlu", x)
    
    assert torch.allclose(bang_ans.cpu(), cpu_ans, atol=1e-3, rtol=1e-3)
    print("Test passed!")
    
    lib.destroyReformDescriptor(descriptor)
    

if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    lib.createReformDescriptor.restype = c_void_p
    lib.destroyReformDescriptor.argtypes = [c_void_p]
    lib.reform.argtypes = [
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
