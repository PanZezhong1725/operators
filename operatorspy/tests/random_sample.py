from ctypes import c_float, c_void_p, c_int
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    CTensor,
    DeviceEnum,
)

from operatorspy.tests.test_utils import get_args
import torch

def random_sample(data, indices, rad, topp, topk, voc):
    p = rad * min(topp * data[voc - 1], data[topk - 1])
    for i in range(voc):
        if(data[i] >= p):
            return indices[i];
    


def test(lib, descriptor, torch_device):
    voc = 10
    data = torch.rand((voc), dtype=torch.float16).to(torch_device)
    #data = torch.tensor(np.arange(voc), dtype=torch.float16).to(torch_device)
    indices = torch.tensor(np.arange(voc), dtype = torch.int32).to(torch_device)
    index = torch.zeros([1], dtype = torch.int32).to(torch_device)
    rad = 0.7
    topp = 0.9
    topk = 5
    
    ans = random_sample(data, indices, rad, topp, topk, voc)
    lib.random_sample(descriptor, to_tensor(data, lib), to_tensor(indices, lib), to_tensor(index, lib), rad, topp, topk, None)
    print(ans)
    print(index)
    assert torch.allclose(index, ans, atol=1e-3, rtol=1e-3)
    print("Test passed!")


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    descriptor = lib.createRandomSampleDescriptor(device, None)
    test(lib, descriptor, "cpu")
    lib.destroyRandomSampleDescriptor(descriptor)


def test_cuda(lib):
    device = DeviceEnum.DEVICE_CUDA

    descriptor = lib.createRandomSampleDescriptor(device, None)
    test(lib, descriptor, "cuda")
    lib.destroyRandomSampleDescriptor(descriptor)


def test_bang(lib):
    import torch_mlu
    device = DeviceEnum.DEVICE_BANG
    descriptor = lib.createRandomSampleDescriptor(device, None)
    test(lib, descriptor, "mlu")
    lib.destroyRandomSampleDescriptor(descriptor)


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    lib.createRandomSampleDescriptor.restype = c_void_p
    lib.destroyRandomSampleDescriptor.argtypes = [c_void_p]
    lib.random_sample.argtypes = [
        c_void_p,
        CTensor,
        CTensor,
        CTensor,
        c_float,
        c_float,
        c_int,
        c_void_p,
    ]
    if args.cpu:
        test_cpu(lib)
    if args.cuda:
        test_cuda(lib)
    if args.bang:
        test_bang(lib)
