#include "unary.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateUnaryDescriptor(CudaHandle_t handle,
                                           UnaryCudaDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t y,
                                           infiniopTensorDescriptor_t x,
                                           int mode) {
    uint64_t ndim = y->ndim;
    if (ndim != x->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!std::equal(y->shape, y->shape + ndim, x->shape)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (mode < 0 || mode >= UnaryMode::numUnaryMode) {
        return STATUS_BAD_PARAM;
    }
    // bitwise operations are only valid for integral types
    if (y->dt.exponent != 0 && mode == UnaryMode::BitwiseNot) {
        return STATUS_BAD_PARAM;
    }

    uint64_t data_size = std::accumulate(y->shape, y->shape + ndim, 1ULL, std::multiplies<uint64_t>());

    *desc_ptr = new UnaryCudaDescriptor{
        DevNvGpu,
        y->dt,
        handle->device_id,
        data_size,
        static_cast<uint64_t>(handle->prop.maxGridSize[0]),
        mode,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyUnaryDescriptor(UnaryCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}