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


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[0], x.shape[-1])
    shape = [d if i == 0 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def rotary_embedding(t, pos, theta, torch_device):
    dh = t.shape[2]
    freqs = (1.0 / (theta ** (torch.arange(0, dh, 2)[: (dh // 2)].float() / dh))).to(
        torch_device
    )
    freqs = torch.outer(pos, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    t_ = torch.view_as_complex(t.reshape(*t.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, t_)
    t_out = torch.view_as_real(t_ * freqs_cis).flatten(2).to(t.dtype)
    return t_out


def test(lib, descriptor, torch_device):
    #t = torch.ones((1, 4, 8), dtype=torch.float16)
    t = torch.zeros(32, dtype = torch.float16)
    for i in range(32):
        t[i] = i
    t = t.reshape([1, 4, 8])
    #pos = torch.ones((1,), dtype=torch.int32)
    pos = torch.ones([1], dtype=torch.int32)
    theta = 1e4
    ans = rotary_embedding(t, pos, theta, "cpu")
    t = t.to(torch_device)
    pos = pos.to(torch_device)

    
    print(ans.cpu())
    lib.rotaryEmbedding(
        descriptor, to_tensor(t, lib), to_tensor(pos, lib), theta, None
    )
    print(t.cpu())
    assert torch.allclose(t.cpu(), ans, atol=1e-3, rtol=1e-3)
    print("Test passed!")


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    config = None
    descriptor = lib.createRotaryEmbeddingDescriptor(device, config)
    test(lib, descriptor, "cpu")
    lib.destroyRotaryEmbeddingDescriptor(descriptor)


def test_cuda(lib):
    device = DeviceEnum.DEVICE_CUDA
    config = None
    descriptor = lib.createRotaryEmbeddingDescriptor(device, config)
    test(lib, descriptor, "cuda")
    lib.destroyRotaryEmbeddingDescriptor(descriptor)

def test_bang(lib):
    import torch_mlu
    device = DeviceEnum.DEVICE_BANG
    config = None
    descriptor = lib.createRotaryEmbeddingDescriptor(device, config)
    test(lib, descriptor, "mlu")
    lib.destroyRotaryEmbeddingDescriptor(descriptor)

if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    lib.createRotaryEmbeddingDescriptor.restype = c_void_p
    lib.destroyRotaryEmbeddingDescriptor.argtypes = [c_void_p]
    lib.rotaryEmbedding.argtypes = [
        c_void_p,
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
