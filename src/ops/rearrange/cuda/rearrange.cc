#include "rearrange.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include <numeric>

infiniopStatus_t cudaCreateRearrangeDescriptor(CudaHandle_t handle,
                                               RearrangeCudaDescriptor_t *desc_ptr,
                                               infiniopTensorDescriptor_t dst,
                                               infiniopTensorDescriptor_t src) {
    auto dt = dst->dt;
    if (!dtype_eq(src->dt, dt)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    auto ndim = dst->ndim;
    if (src->ndim != ndim || ndim == 0) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (int i = 0; i < ndim; ++i) {
        if (dst->shape[i] != src->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (dst->strides[ndim - 1] != 1 || src->strides[ndim - 1] != 1) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    switch (ndim) {
        case 1:
            *desc_ptr = new RearrangeCudaDescriptor{
                handle->device,
                handle->device_id,
                dt.size * dst->shape[0],
                1, 1,
                0, 0,
                0, 0};
            break;
        case 2:
            *desc_ptr = new RearrangeCudaDescriptor{
                handle->device,
                handle->device_id,
                dt.size * dst->shape[1],
                1, dst->shape[0],
                0, dst->strides[0],
                0, src->strides[0]};
            break;
        case 3:
            *desc_ptr = new RearrangeCudaDescriptor{
                handle->device,
                handle->device_id,
                dt.size * dst->shape[2],
                dst->shape[0], dst->shape[1],
                dst->strides[0], dst->strides[1],
                src->strides[0], src->strides[1]};
            break;
        default:
            return STATUS_BAD_TENSOR_SHAPE;
    }

    (*desc_ptr)->dst_rs *= dt.size;
    (*desc_ptr)->dst_cs *= dt.size;
    (*desc_ptr)->src_rs *= dt.size;
    (*desc_ptr)->src_cs *= dt.size;

    return STATUS_SUCCESS;
}
infiniopStatus_t cudaDestroyRearrangeDescriptor(RearrangeCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
