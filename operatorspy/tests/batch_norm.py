from ctypes import POINTER, Structure, c_int32, c_void_p, c_double
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
    device_enum_to_str,
)

from operatorspy.tests.test_utils import get_args
from enum import Enum, auto
import torch
import ctypes
import torch.nn.functional as F

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


class BatchNormDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopBatchNormDescriptor_t = POINTER(BatchNormDescriptor)


def batch_norm(x, scale, b, mean, var, eps):
    ndim = len(x.shape)
    if ndim <= 1 or ndim > 5:
        print("Error: Pytorch -> Unsupported tensor dimension")
        return None
    if PROFILE:
        ans = F.batch_norm(x, mean, var, scale, b, training=False, eps=eps)
        torch.cuda.synchronize()
        return ans
    return F.batch_norm(x, mean, var, scale, b, training=False, eps=eps)


# get the mean and variance of the input tensor across the batch size N and spatial dimensions
def get_mean_variance(x, dtype):
    dims = tuple(range(x.ndim))
    reduction_dims = tuple(d for d in dims if d != 1)  # Exclude the channel dimension
    return x.mean(dim=reduction_dims, dtype=dtype), x.var(
        dim=reduction_dims, unbiased=False
    ).to(dtype)


def test(
    lib,
    handle,
    torch_device,
    x_shape,
    eps=1e-5,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing BatchNorm on {torch_device} with x_shape: {x_shape}, scale_shape: {x_shape[1]}, b_shape: {x_shape[1]}, mean_shape: {x_shape[1]}, var_shape: {x_shape[1]}, eps: {eps}, dtype:{tensor_dtype}, Inplace:{inplace}"
    )
    num_channel = x_shape[1]
    bn_dtype = tensor_dtype if tensor_dtype != torch.float16 else torch.float32
    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device) * 10 - 2
    scale = torch.rand(num_channel, dtype=bn_dtype).to(torch_device)
    b = torch.rand(num_channel, dtype=bn_dtype).to(torch_device)
    mean, var = get_mean_variance(x, bn_dtype)
    y = torch.zeros(x_shape, dtype=tensor_dtype).to(torch_device) if inplace == Inplace.OUT_OF_PLACE else x

    # get the pytorch answer
    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = batch_norm(x, scale, b, mean, var, eps)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = batch_norm(x, scale, b, mean, var, eps)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    # get the operators' answer
    x_tensor = to_tensor(x, lib)
    scale_tensor = to_tensor(scale, lib)
    b_tensor = to_tensor(b, lib)
    mean_tensor = to_tensor(mean, lib)
    var_tensor = to_tensor(var, lib)
    y_tensor = to_tensor(y, lib) if inplace == Inplace.OUT_OF_PLACE else x_tensor
    descriptor = infiniopBatchNormDescriptor_t()

    check_error(
        lib.infiniopCreateBatchNormDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            scale_tensor.descriptor,
            b_tensor.descriptor,
            mean_tensor.descriptor,
            var_tensor.descriptor,
            eps,
        )
    )

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopBatchNorm(
                descriptor,
                y_tensor.data,
                x_tensor.data,
                scale_tensor.data,
                b_tensor.data,
                mean_tensor.data,
                var_tensor.data,
                None,
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopBatchNorm(
                    descriptor,
                    y_tensor.data,
                    x_tensor.data,
                    scale_tensor.data,
                    b_tensor.data,
                    mean_tensor.data,
                    var_tensor.data,
                    None,
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")

    if (tensor_dtype == torch.float16):
        assert torch.allclose(y, ans, atol=1e-5, rtol=1e-3)
    else: # float32
        assert torch.allclose(y, ans, atol=1e-6, rtol=1e-5)
    check_error(lib.infiniopDestroyBatchNormDescriptor(descriptor))


def test_operator(lib, device, test_cases, tensor_dtypes):
    handle = create_handle(lib, device)
    for x_shape, eps, inplace in test_cases:
        for tensor_dtype in tensor_dtypes:
            test(lib, handle, device_enum_to_str(device), x_shape, eps, inplace=inplace, tensor_dtype=tensor_dtype)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, eps, inplace
        ((2, 5, 7), 1e-5, Inplace.OUT_OF_PLACE),
        ((2, 5, 7), 1e-5, Inplace.INPLACE_X),
        ((32, 3, 1024), 1e-5, Inplace.OUT_OF_PLACE),
        ((32, 3, 128, 128), 1e-5, Inplace.OUT_OF_PLACE),
        ((32, 3, 64, 64, 64), 1e-5, Inplace.OUT_OF_PLACE),
    ]
    tensor_dtypes = [
        torch.float16, torch.float32,
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateBatchNormDescriptor.restype = c_int32
    lib.infiniopCreateBatchNormDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopBatchNormDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_double,
    ]
    lib.infiniopBatchNorm.restype = c_int32
    lib.infiniopBatchNorm.argtypes = [
        infiniopBatchNormDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyBatchNormDescriptor.restype = c_int32
    lib.infiniopDestroyBatchNormDescriptor.argtypes = [
        infiniopBatchNormDescriptor_t,
    ]

    if args.cpu:
        test_operator(lib, DeviceEnum.DEVICE_CPU, test_cases, tensor_dtypes)
    if args.cuda:
        test_operator(lib, DeviceEnum.DEVICE_CUDA, test_cases, tensor_dtypes)
    if args.bang:
        import torch_mlu
        test_operator(lib, DeviceEnum.DEVICE_BANG, test_cases, tensor_dtypes)
    if not (args.cpu or args.cuda or args.bang):
        test_operator(lib, DeviceEnum.DEVICE_CPU, test_cases, tensor_dtypes)
    print("\033[92mTest passed!\033[0m")
