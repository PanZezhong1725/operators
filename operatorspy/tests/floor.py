from ctypes import POINTER, Structure, c_int32, c_void_p
import ctypes
import sys
import os
import time

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

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


class FloorDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopFloorDescriptor_t = POINTER(FloorDescriptor)

def find_and_print_differing_indices(
    x, tensor1, tensor2, atol=0, rtol=1e-2
):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape to compare.")

    # Calculate the difference mask based on atol and rtol
    diff_mask = torch.abs(tensor1 - tensor2) > (atol + rtol * torch.abs(tensor2))
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)

    # Print the indices and the differing elements
    for idx in diff_indices:
        index_tuple = tuple(idx.tolist())
        print(
            f"Index: {index_tuple}, x: {x[index_tuple]}, y element: {tensor1[index_tuple]}, ans element: {tensor2[index_tuple]}"
        )

    return diff_indices


def floor(x):
    if PROFILE:
        ans = torch.floor(x).to(x.dtype)
        torch.cuda.synchronize()
        return ans
    return torch.floor(x).to(x.dtype)


def test(
    lib,
    handle,
    torch_device,
    tensor_shape, 
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Floor on {torch_device} with tensor_shape:{tensor_shape} dtype:{tensor_dtype} inplace: {inplace.name}"
    )

    x = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device) * 10 - 20
    y = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device) if inplace == Inplace.OUT_OF_PLACE else x

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = floor(x)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = floor(x)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib) if inplace == Inplace.OUT_OF_PLACE else x_tensor
    descriptor = infiniopFloorDescriptor_t()

    check_error(
        lib.infiniopCreateFloorDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
        )
    )
    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(lib.infiniopFloor(descriptor, y_tensor.data, x_tensor.data, None))
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopFloor(descriptor, y_tensor.data, x_tensor.data, None)
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")

    assert torch.allclose(y, ans, atol=0, rtol=1e-3, equal_nan=True)
    check_error(lib.infiniopDestroyFloorDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for tensor_shape, inplace in test_cases:
        test(lib, handle, "cpu", tensor_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cpu", tensor_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for tensor_shape, inplace in test_cases:
        test(lib, handle, "cuda", tensor_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cuda", tensor_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for tensor_shape, inplace in test_cases:
        test(lib, handle, "mlu", tensor_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "mlu", tensor_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # tensor_shape, inplace
        ((), Inplace.OUT_OF_PLACE),
        ((), Inplace.INPLACE_X),
        ((1, 3), Inplace.OUT_OF_PLACE),
        ((3, 3), Inplace.OUT_OF_PLACE),
        ((3, 3, 13, 9, 17), Inplace.INPLACE_X),
        ((32, 20, 512), Inplace.INPLACE_X),
        ((33, 333, 333), Inplace.OUT_OF_PLACE),
        ((32, 256, 112, 112), Inplace.OUT_OF_PLACE),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateFloorDescriptor.restype = c_int32
    lib.infiniopCreateFloorDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopFloorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopFloor.restype = c_int32
    lib.infiniopFloor.argtypes = [
        infiniopFloorDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyFloorDescriptor.restype = c_int32
    lib.infiniopDestroyFloorDescriptor.argtypes = [
        infiniopFloorDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
