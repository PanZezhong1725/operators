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
)

from operatorspy.tests.test_utils import get_args
import torch
import ctypes
import torch.nn.functional as F
import numpy as np

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


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


def find_and_print_differing_indices(
    x, tensor1, tensor2, mean, scale, var, b, atol=0, rtol=1e-2
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
            f"Index: {index_tuple}, x: {x[index_tuple]}, mean: {mean[index_tuple[1]]}, scale: {scale[index_tuple[1]]}, var: {var[index_tuple[1]]}, b: {b[index_tuple[1]]}, y element: {tensor1[index_tuple]}, ans element: {tensor2[index_tuple]}"
        )

    return diff_indices


def test(
    lib,
    handle,
    torch_device,
    x_shape,
    eps=1e-5,
    tensor_dtype=torch.float16,
):
    print(
        f"Testing BatchNorm on {torch_device} with x_shape: {x_shape}, scale_shape: {x_shape[1]}, b_shape: {x_shape[1]}, mean_shape: {x_shape[1]}, var_shape: {x_shape[1]}, eps: {eps} dtype:{tensor_dtype}"
    )
    num_channel = x_shape[1]
    bn_dtype = tensor_dtype if tensor_dtype != torch.float16 else torch.float32
    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device) * 10 - 2
    scale = torch.rand(num_channel, dtype=bn_dtype).to(torch_device)
    b = torch.rand(num_channel, dtype=bn_dtype).to(torch_device)
    mean, var = get_mean_variance(x, bn_dtype)
    y = torch.zeros(x_shape, dtype=tensor_dtype).to(torch_device)

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
    y_tensor = to_tensor(y, lib)
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
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")

    # print(" - x: \n", x, "\n - y:\n", y, "\n - ans:\n", ans)
    # print(" - y:\n", y, "\n - ans:\n", ans)

    # find_and_print_differing_indices(x, y, ans, mean, scale, mean, b, atol=1e-7, rtol=1e-3)
    # np.testing.assert_allclose(y.numpy(), ans.numpy(), atol=1e-7, rtol=1e-3)
    assert torch.allclose(y, ans, atol=1e-7, rtol=1e-3)
    check_error(lib.infiniopDestroyBatchNormDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, eps in test_cases:
        test(lib, handle, "cpu", x_shape, eps, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", x_shape, eps, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape, eps in test_cases:
        test(lib, handle, "cuda", x_shape, eps, tensor_dtype=torch.float16)
        test(lib, handle, "cuda", x_shape, eps, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for x_shape, eps in test_cases:
        test(lib, handle, "mlu", x_shape, eps, tensor_dtype=torch.float16)
        test(lib, handle, "mlu", x_shape, eps, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, eps
        ((2, 3, 4), 1e-5),
        ((32, 3, 1024), 1e-5),
        ((1, 3, 4, 4), 1e-5),
        ((32, 3, 128, 128), 1e-5),
        ((1, 6, 5, 5, 5), 1e-5),
        ((32, 3, 64, 64, 64), 1e-5),
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
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
