from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64, c_int64
import ctypes
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
from enum import Enum, auto
import torch


class Inplace(Enum):
    OUT_OF_PLACE = auto()

class ConcatDescriptor(Structure):
    _fields_ = [("device", c_int32),]


infiniopConcatDescriptor_t = POINTER(ConcatDescriptor)


def concat_py(*tensors, dim=0):
    return torch.cat(tensors, dim=dim)


def test(
    lib,
    handle,
    torch_device,
    c_shape,
    axis,
    input_shapes,
    tensor_dtype=torch.float32,
    inplace=Inplace.OUT_OF_PLACE,
):
    """
    测试 concat 算子
    """
    print(
        f"Testing Concat on {torch_device} with output_shape:{c_shape}, input_shapes:{input_shapes}, axis:{axis}, dtype:{tensor_dtype}, inplace: {inplace.name}"
    )
    
    inputs = [torch.rand(shape, dtype=tensor_dtype).to(torch_device) for shape in input_shapes]
    
    if inplace == Inplace.OUT_OF_PLACE:
        c = torch.zeros(c_shape, dtype=tensor_dtype).to(torch_device)
    else:
        c = torch.zeros(c_shape, dtype=tensor_dtype).to(torch_device)
    
    ans = concat_py(*inputs, dim=axis)
    
    input_tensors = [to_tensor(t, lib) for t in inputs]
    c_tensor = to_tensor(c, lib) if inplace == Inplace.OUT_OF_PLACE else to_tensor(c, lib)
    
    descriptor = infiniopConcatDescriptor_t()

    num_inputs = len(input_tensors)
    input_desc_array_type = infiniopTensorDescriptor_t * num_inputs
    input_desc_array = input_desc_array_type(*[t.descriptor for t in input_tensors])
    
    check_error(
        lib.infiniopCreateConcatDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,            
            input_desc_array,              
            c_uint64(num_inputs),
            c_int64(axis),
        )
    )

    input_data_ptrs = (c_void_p * num_inputs)(*[t.data for t in input_tensors])
    check_error(
        lib.infiniopConcat(
            descriptor,
            c_tensor.data,
            ctypes.cast(input_data_ptrs, POINTER(c_void_p)),
            None  
        )
    )

    assert torch.allclose(c, ans, atol=0, rtol=0), "Concat result does not match PyTorch's result."
    
    check_error(lib.infiniopDestroyConcatDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for c_shape, axis, input_shapes, inplace in test_cases:
        test(lib, handle, "cpu", c_shape, axis, input_shapes, inplace=inplace)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for c_shape, axis, input_shapes, inplace in test_cases:
        test(lib, handle, "cuda", c_shape, axis, input_shapes, inplace=inplace)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu
    
    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for c_shape, axis, input_shapes, inplace in test_cases:
        test(lib, handle, "mlu", c_shape, axis, input_shapes, inplace=inplace)
    destroy_handle(lib, handle)


if __name__ == "__main__":

    test_cases = [

        ((6,), 0, [(2,), (4,)], Inplace.OUT_OF_PLACE),  

        ((6, 3), 0, [(2, 3), (4, 3)], Inplace.OUT_OF_PLACE),  
        ((3, 6), 1, [(3, 2), (3, 4)], Inplace.OUT_OF_PLACE),  
        ((3, 7), 1, [(3, 2), (3, 4), (3, 1)], Inplace.OUT_OF_PLACE), 
        ((3, 3, 10), 2, [(3, 3, 4), (3, 3, 6)], Inplace.OUT_OF_PLACE),  

        ((4, 3, 6), 0, [(3, 3, 6), (1, 3, 6)], Inplace.OUT_OF_PLACE),  
        ((2, 6, 3), 1, [(2, 3, 3), (2, 3, 3)], Inplace.OUT_OF_PLACE),  
        ((2, 3, 6), 2, [(2, 3, 3), (2, 3, 3)], Inplace.OUT_OF_PLACE),  

        ((4, 3, 5, 6), 0, [(1, 3, 5, 6), (3, 3, 5, 6)], Inplace.OUT_OF_PLACE),  
        ((2, 5, 5, 6), 1, [(2, 3, 5, 6), (2, 2, 5, 6)], Inplace.OUT_OF_PLACE),  
        ((2, 3, 5, 6), 2, [(2, 3, 2, 6), (2, 3, 3, 6)], Inplace.OUT_OF_PLACE),  
        ((2, 3, 5, 6), 3, [(2, 3, 5, 3), (2, 3, 5, 3)], Inplace.OUT_OF_PLACE),  
        ((2, 3, 5, 15), 3, [(2, 3, 5, 3), (2, 3, 5, 3), (2, 3, 5, 9)], Inplace.OUT_OF_PLACE),  

        ((4, 2, 3, 4, 5), 0, [(1, 2, 3, 4, 5), (3, 2, 3, 4, 5)], Inplace.OUT_OF_PLACE),  
        ((2, 4, 3, 2, 5), 1, [(2, 2, 3, 2, 5), (2, 2, 3, 2, 5)], Inplace.OUT_OF_PLACE),  
        ((1, 2, 4, 4, 5), 2, [(1, 2, 2, 4, 5), (1, 2, 2, 4, 5)], Inplace.OUT_OF_PLACE),  
        ((1, 2, 3, 8, 5), 3, [(1, 2, 3, 4, 5), (1, 2, 3, 4, 5)], Inplace.OUT_OF_PLACE), 
        ((1, 2, 3, 4, 5), 4, [(1, 2, 3, 4, 3), (1, 2, 3, 4, 2)], Inplace.OUT_OF_PLACE),  
        ((4, 14, 3, 4, 5), 1, [(4, 3, 3, 4, 5), (4, 5, 3, 4, 5), (4, 6, 3, 4, 5)], Inplace.OUT_OF_PLACE),  


        ((6,), -1, [(2,), (4,)], Inplace.OUT_OF_PLACE),  

        ((6, 3), -2, [(2, 3), (4, 3)], Inplace.OUT_OF_PLACE),  
        ((3, 6), -1, [(3, 2), (3, 4)], Inplace.OUT_OF_PLACE),  
        ((3, 7), -1, [(3, 2), (3, 4), (3, 1)], Inplace.OUT_OF_PLACE), 
        ((3, 3, 10), -1, [(3, 3, 4), (3, 3, 6)], Inplace.OUT_OF_PLACE),  

        ((4, 3, 6), -3, [(3, 3, 6), (1, 3, 6)], Inplace.OUT_OF_PLACE),  
        ((2, 6, 3), -2, [(2, 3, 3), (2, 3, 3)], Inplace.OUT_OF_PLACE),  
        ((2, 3, 6), -1, [(2, 3, 3), (2, 3, 3)], Inplace.OUT_OF_PLACE),  

        ((4, 3, 5, 6), -4, [(1, 3, 5, 6), (3, 3, 5, 6)], Inplace.OUT_OF_PLACE),  
        ((2, 5, 5, 6), -3, [(2, 3, 5, 6), (2, 2, 5, 6)], Inplace.OUT_OF_PLACE),  
        ((2, 3, 5, 6), -2, [(2, 3, 2, 6), (2, 3, 3, 6)], Inplace.OUT_OF_PLACE),  
        ((2, 3, 5, 6), -1, [(2, 3, 5, 3), (2, 3, 5, 3)], Inplace.OUT_OF_PLACE),  
        ((2, 3, 5, 15), -1, [(2, 3, 5, 3), (2, 3, 5, 3), (2, 3, 5, 9)], Inplace.OUT_OF_PLACE),  

        ((4, 2, 3, 4, 5), -5, [(1, 2, 3, 4, 5), (3, 2, 3, 4, 5)], Inplace.OUT_OF_PLACE),  
        ((2, 4, 3, 2, 5), -4, [(2, 2, 3, 2, 5), (2, 2, 3, 2, 5)], Inplace.OUT_OF_PLACE),  
        ((1, 2, 4, 4, 5), -3, [(1, 2, 2, 4, 5), (1, 2, 2, 4, 5)], Inplace.OUT_OF_PLACE),  
        ((1, 2, 3, 8, 5), -2, [(1, 2, 3, 4, 5), (1, 2, 3, 4, 5)], Inplace.OUT_OF_PLACE), 
        ((1, 2, 3, 4, 5), -1, [(1, 2, 3, 4, 3), (1, 2, 3, 4, 2)], Inplace.OUT_OF_PLACE),  
        ((4, 14, 3, 4, 5), -4, [(4, 3, 3, 4, 5), (4, 5, 3, 4, 5), (4, 6, 3, 4, 5)], Inplace.OUT_OF_PLACE),     
    ]
    
    args = get_args()
    lib = open_lib()
    
    lib.infiniopCreateConcatDescriptor.restype = c_int32
    lib.infiniopCreateConcatDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopConcatDescriptor_t),
        infiniopTensorDescriptor_t,  
        POINTER(infiniopTensorDescriptor_t),  
        c_uint64,  # nums_input
        c_int64,  # axis
    ]
    
    lib.infiniopConcat.restype = c_int32
    lib.infiniopConcat.argtypes = [
        infiniopConcatDescriptor_t,
        c_void_p,  
        POINTER(c_void_p),  
        c_void_p, 
    ]
    
    lib.infiniopDestroyConcatDescriptor.restype = c_int32
    lib.infiniopDestroyConcatDescriptor.argtypes = [
        infiniopConcatDescriptor_t,
    ]
    
    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    
    print("\033[92mConcat Test passed!\033[0m")




