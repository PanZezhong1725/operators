#include "pooling_cpu.h"
#include "../../utils.h"
#include <cstring>
#include <numeric>

infiniopStatus_t cpuCreatePoolingDescriptor(infiniopHandle_t,
                                            PoolingCpuDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x,
                                            void const *kernel_shape,
                                            void const *pads,
                                            void const *strides,
                                            uint64_t n,
                                            int pooling_type) {
    uint64_t ndim = y->ndim;
    if (ndim < 3 || ndim != x->ndim || ndim != n + 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (x->shape[0] != y->shape[0] || x->shape[1] != y->shape[1]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (pooling_type > 1) {
        return STATUS_BAD_PARAM;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    *desc_ptr = new PoolingCpuDescriptor{
        DevCpu,
        y->dt,
        ndim,
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyPoolingDescriptor(PoolingCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuPooling(PoolingCpuDescriptor_t desc,
                            void *y,
                            void const *x,
                            void *stream) {
    return STATUS_SUCCESS;
}
