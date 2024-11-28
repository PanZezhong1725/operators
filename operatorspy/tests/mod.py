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
# NOTE: need to manually add synchronization function to the lib function (elementwise.cu),
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()
    INPLACE_AB = auto()


class ModDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopModDescriptor_t = POINTER(ModDescriptor)


def find_and_print_differing_indices(a, b, c, ans, atol=0, rtol=1e-2):
    if c.shape != ans.shape:
        raise ValueError("Tensors must have the same shape to compare.")

    # Calculate the difference mask based on atol and rtol
    diff_mask = torch.abs(c - ans) > (atol + rtol * torch.abs(ans))
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)

    # Print the indices and the differing elements
    for idx in diff_indices:
        index_tuple = tuple(idx.tolist())
        print(
            f"Index: {index_tuple}, a: {a[index_tuple]}, b: {b[index_tuple]}, y element: {c[index_tuple]}, ans element: {ans[index_tuple]}"
        )

    return diff_indices


def mod(x, y):
    if PROFILE:
        ans = torch.fmod(x, y)
        torch.cuda.synchronize()
        return ans
    return torch.fmod(x, y)


def test(
    lib,
    handle,
    torch_device,
    c_shape,
    a_shape,
    b_shape,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Mod on {torch_device} with c_shape:{c_shape} a_shape:{a_shape} b_shape:{b_shape} dtype:{tensor_dtype} inplace: {inplace.name}"
    )
    if a_shape != b_shape and inplace != Inplace.OUT_OF_PLACE:
        print("Unsupported test: broadcasting does not support in-place")
        return

    a = torch.rand(a_shape, dtype=tensor_dtype).to(torch_device) * 10
    b = (
        torch.rand(b_shape, dtype=tensor_dtype).to(torch_device)
        if inplace != Inplace.INPLACE_AB
        else a
    )
    c = (
        torch.rand(c_shape, dtype=tensor_dtype).to(torch_device)
        if inplace == Inplace.OUT_OF_PLACE
        else (a if inplace == Inplace.INPLACE_A else b)
    )

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = mod(a, b)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = mod(a, b)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    a_tensor = to_tensor(a, lib)
    b_tensor = to_tensor(b, lib) if inplace != Inplace.INPLACE_AB else a_tensor
    c_tensor = (
        to_tensor(c, lib)
        if inplace == Inplace.OUT_OF_PLACE
        else (a_tensor if inplace == Inplace.INPLACE_A else b_tensor)
    )
    descriptor = infiniopModDescriptor_t()

    check_error(
        lib.infiniopCreateModDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopMod(
                descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopMod(
                    descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")

    assert torch.allclose(c, ans, atol=0, rtol=0, equal_nan=True)
    check_error(lib.infiniopDestroyModDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        test(
            lib,
            handle,
            "cpu",
            c_shape,
            a_shape,
            b_shape,
            tensor_dtype=torch.float16,
            inplace=inplace,
        )
        test(
            lib,
            handle,
            "cpu",
            c_shape,
            a_shape,
            b_shape,
            tensor_dtype=torch.float32,
            inplace=inplace,
        )
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        test(
            lib,
            handle,
            "cuda",
            c_shape,
            a_shape,
            b_shape,
            tensor_dtype=torch.float16,
            inplace=inplace,
        )
        test(
            lib,
            handle,
            "cuda",
            c_shape,
            a_shape,
            b_shape,
            tensor_dtype=torch.float32,
            inplace=inplace,
        )
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        test(
            lib,
            handle,
            "mlu",
            c_shape,
            a_shape,
            b_shape,
            tensor_dtype=torch.float16,
            inplace=inplace,
        )
        test(
            lib,
            handle,
            "mlu",
            c_shape,
            a_shape,
            b_shape,
            tensor_dtype=torch.float32,
            inplace=inplace,
        )
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # c_shape, a_shape, b_shape, inplace
        # ((32, 150, 51200), (32, 150, 51200), (32, 150, 1), Inplace.OUT_OF_PLACE),
        ((1, 3), (1, 3), (1, 3), Inplace.OUT_OF_PLACE),
        ((), (), (), Inplace.OUT_OF_PLACE),
        ((2, 4, 3), (2, 1, 3), (4, 3), Inplace.OUT_OF_PLACE),
        ((2, 3, 4, 5), (2, 3, 4, 5), (5,), Inplace.OUT_OF_PLACE),
        ((3, 2, 4, 5), (4, 5), (3, 2, 1, 1), Inplace.OUT_OF_PLACE),
        ((3, 20, 33), (3, 20, 33), (3, 20, 33), Inplace.OUT_OF_PLACE),
        (
            ((3, 20, 33), (3, 20, 33), (3, 20, 33), Inplace.INPLACE_A)
            if not PROFILE
            else ((32, 10, 100), (32, 10, 100), (32, 10, 100), Inplace.OUT_OF_PLACE)
        ),
        (
            ((3, 20, 33), (3, 20, 33), (3, 20, 33), Inplace.INPLACE_B)
            if not PROFILE
            else ((32, 15, 510), (32, 15, 510), (32, 15, 510), Inplace.OUT_OF_PLACE)
        ),
        (
            ((3, 20, 33), (3, 20, 33), (3, 20, 33), Inplace.INPLACE_AB)
            if not PROFILE
            else (
                (32, 256, 112, 112),
                (32, 256, 112, 1),
                (32, 256, 112, 112),
                Inplace.OUT_OF_PLACE,
            )
        ),
        (
            (
                (32, 3, 112, 112),
                (32, 3, 112, 112),
                (32, 3, 112, 112),
                Inplace.OUT_OF_PLACE,
            )
            if not PROFILE
            else (
                (32, 256, 112, 112),
                (32, 256, 112, 112),
                (32, 256, 112, 112),
                Inplace.OUT_OF_PLACE,
            )
        ),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateModDescriptor.restype = c_int32
    lib.infiniopCreateModDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopModDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopMod.restype = c_int32
    lib.infiniopMod.argtypes = [
        infiniopModDescriptor_t,
   