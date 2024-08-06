import os
import platform
import ctypes
from ctypes import c_void_p, c_int, c_int64, c_uint64, Structure, POINTER
from .data_layout import *

Device = c_int
Optype = c_int

LIB_OPERATORS_DIR = "INFINI_ROOT"


class TensorLayout(Structure):
    _fields_ = [
        ("dt", DataLayout),
        ("ndim", c_uint64),
        ("shape", POINTER(c_uint64)),
        ("pattern", POINTER(c_int64)),
    ]


TensorDescriptor = ctypes.POINTER(TensorLayout)


class CTensor(Structure):
    _fields_ = [("layout", TensorDescriptor), ("data", c_void_p)]


# Open operators library
def open_lib():
    def find_library_in_ld_path(library_name):
        ld_library_path = os.environ.get(LIB_OPERATORS_DIR, "")
        paths = ld_library_path.split(os.pathsep)
        for path in paths:
            full_path = os.path.join(path, library_name)
            if os.path.isfile(full_path):
                return full_path
        return None

    system_name = platform.system()
    # Load the library
    if system_name == 'Windows':
        library_path = find_library_in_ld_path("operators.dll")
    elif system_name == 'Linux':
        library_path = find_library_in_ld_path("liboperators.so")

    assert (
        library_path is not None
    ), f"Cannot find operators.dll or liboperators.so. Check if {LIB_OPERATORS_DIR} is set correctly."
    lib = ctypes.CDLL(library_path)
    lib.createTensorDescriptor.argtypes = [
        POINTER(POINTER(TensorLayout)),
        c_uint64,
        POINTER(c_uint64),
        POINTER(c_int64),
        DataLayout,
    ]
    return lib


# Convert PyTorch tensor to library Tensor
def to_tensor(tensor, lib, shape = None, strides = None):
    import torch

    ndim = tensor.ndimension()
    if shape is None:
        shape = (ctypes.c_uint64 * ndim)(*tensor.shape)
    else:
        shape = (ctypes.c_uint64 * ndim)(*shape)
    # Get strides in bytes
    if strides is None:
        strides = (ctypes.c_int64 * ndim)(
            *(s * tensor.element_size() for s in tensor.stride())
        )
    else:
        strides = (ctypes.c_int64 * ndim)(*strides)
    data_ptr = tensor.data_ptr()
    # fmt: off
    dt = (
        I8 if tensor.dtype == torch.int8 else
        I16 if tensor.dtype == torch.int16 else
        I32 if tensor.dtype == torch.int32 else
        I64 if tensor.dtype == torch.int64 else
        U8 if tensor.dtype == torch.uint8 else
        F16 if tensor.dtype == torch.float16 else
        BF16 if tensor.dtype == torch.bfloat16 else
        F32 if tensor.dtype == torch.float32 else
        F64 if tensor.dtype == torch.float64 else
        # TODO: These following types may not be supported by older 
        # versions of PyTorch.
        U16 if tensor.dtype == torch.uint16 else
        U32 if tensor.dtype == torch.uint32 else
        U64 if tensor.dtype == torch.uint64 else
        None
    )
    # fmt: on
    assert dt is not None
    # Create TensorDecriptor
    tensor_desc = TensorDescriptor()
    lib.createTensorDescriptor(ctypes.byref(tensor_desc), ndim, shape, strides, dt)
    # Create Tensor
    return CTensor(tensor_desc, ctypes.c_void_p(data_ptr))
