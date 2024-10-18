#include "relu.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateReluDescriptor(CudaHandle_t handle,
                                          ReluCudaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x) {
    uint64_t ndim = y->ndim;
    if (ndim != x->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (y->shape[i] != x->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (y->dt != F16 || y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t data_size = std::accumulate(y->shape, y->shape + y->ndim, 1ULL, std::multiplies<uint64_t>());

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, handle->device_id);

    *desc_ptr = new ReluCudaDescriptor{
        DevNvGpu,
        y->dt,
        handle->device_id,
        ndim,
        data_size,
        static_cast<uint64_t>(prop.maxGridSize[0]),
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyReluDescriptor(ReluCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
