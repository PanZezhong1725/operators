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

def random_sample(data, topp, topk, voc, temperature):
    indices = torch.zeros([topk], dtype = torch.int32)
    dataNp = data.clone().numpy()
    print(dataNp)
    sorted_indices = np.argsort(dataNp)[::-1]  
    indices = sorted_indices[:topk] 
    dataNp = dataNp[sorted_indices]
    print(dataNp)
    print(indices)
    globalM = dataNp[0]
    dataNp = torch.tensor((dataNp - globalM) / temperature)
    dataNp = torch.softmax(dataNp, dim = 0)
    sum_s = 0
    for end in range(topk):
        sum_s += dataNp[end]
        if(sum_s >= topp):
            break
    if(end < topk - 1):
        end += 1
    else:
        end = topk
    
    #rad = torch.rand(1)
    rad = 0.75
    sum_s = 0
    for i in range(end):
        sum_s += dataNp[i]
    rad *= sum_s
    #print(rad)
    sum_s = 0
    for i in range(end):
        sum_s += dataNp[i]
        if(rad < sum_s):
            return torch.tensor(indices[i]).to(torch.int32)
    
    


def test(lib, descriptor, torch_device):
    voc = 10
    data = torch.rand((voc), dtype=torch.float16).to(torch_device)
    #data = torch.tensor(np.arange(voc), dtype=torch.float16).to(torch_device)
    indices = torch.zeros([1], dtype = torch.int32).to(torch_device)
    topp = 0.9
    topk = 3
    temperature = 2.0

    ans = random_sample(data.to("cpu"), topp, topk, voc, temperature)
    lib.random_sample(descriptor, to_tensor(data, lib), to_tensor(indices, lib), topp, topk, temperature, None)
    print(ans)
    print(indices.cpu())
    assert torch.allclose(indices.flatten(), ans, atol=1e-3, rtol=1e-3)
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
        c_float,
        c_int,
        c_float,
        c_void_p,
    ]
    if args.cpu:
        test_cpu(lib)
    if args.cuda:
        test_cuda(lib)
    if args.bang:
        test_bang(lib)
