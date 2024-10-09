#include "rearrange.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../../utils.h"
#include <numeric>

infiniopStatus_t cudaCreateRearrangeDescriptor(CudaHandle_t handle,
                                               RearrangeCudaDescriptor_t *desc_ptr,
                                               infiniopTensorDescriptor_t dst,
                                               infiniopTensorDescriptor_t src) {
    if (!dtype_eq(dst->dt, src->dt)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (dst->ndim != src->ndim || dst->ndim < 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    auto ndim = dst->ndim;
    for (int i = 0; i < ndim; ++i) {
        if (dst->shape[i] != src->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (dst->strides[ndim - 1] != 1 || src->strides[ndim - 1] != 1) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    unsigned int r = 0, c = 0, b = 0;
    unsigned int rsa = 0, csa = 0, rsb = 0, csb = 0;
    if (ndim == 2) {
        c = dst->shape[0];
        b = dst->shape[1];
        csa = dst->strides[0];
        csb = src->strides[0];
    } else if (ndim == 3) {
        r = dst->shape[0];
        c = dst->shape[1];
        b = dst->shape[2];
        csa = dst->strides[1];
        csb = src->strides[1];
        rsa = dst->strides[0];
        rsb = src->strides[0];
    } else {
        for (int i = ndim - 3; i >= 1; --i) {
            if (dst->shape[i] * dst->strides[i] != dst->strides[i - 1] || src->shape[i] * src->strides[i] != src->strides[i - 1]) {
                return STATUS_BAD_TENSOR_STRIDES;
            }
        }
        r = std::accumulate(dst->shape, dst->shape + ndim - 2, 1, std::multiplies<unsigned int>());
        c = dst->shape[ndim - 2];
        b = dst->shape[ndim - 1];
        csa = dst->strides[ndim - 2];
        csb = src->strides[ndim - 2];
        rsa = dst->strides[ndim - 3];
        rsb = src->strides[ndim - 3];
    }
    auto contiguous_bytes = b * dst->dt.size;
    if (contiguous_bytes % WARP_SIZE != 0) {
        return STATUS_BAD_PARAM;
    }
    auto bytes_per_thread = contiguous_bytes / WARP_SIZE;
    if (bytes_per_thread <= 0 || bytes_per_thread > 32 || (bytes_per_thread & (bytes_per_thread - 1)) != 0) {
        return STATUS_BAD_PARAM;
    }
    *desc_ptr = new RearrangeCudaDescriptor{
        handle->device,
		handle->device_id,
        rsa,
        rsb,
        csa,
        csb,
        r, c, b,
        bytes_per_thread};
    return STATUS_SUCCESS;
}
infiniopStatus_t cudaDestroyRearrangeDescriptor(RearrangeCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
