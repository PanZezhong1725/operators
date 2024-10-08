#include "relu_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateReluDescriptor(infiniopHandle_t,
                                         ReluCpuDescriptor_t *desc_ptr,
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

    *desc_ptr = new ReluCpuDescriptor{
        DevCpu,
        y->dt,
        data_size,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyReluDescriptor(ReluCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

void relu_cpu_f16(ReluCpuDescriptor_t desc, void *y, void const *x) {
    auto x_ = reinterpret_cast<uint16_t const *>(x);
    auto y_ = reinterpret_cast<uint16_t *>(y);

    for (uint64_t i = 0; i < desc->data_size; ++i) {
        float x_f32 = f16_to_f32(x_[i]);
        y_[i] = f32_to_f16(x_f32 < 0 ? 0 : x_f32);
    }
}

infiniopStatus_t cpuRelu(ReluCpuDescriptor_t desc,
                         void *y, void const *x,
                         void *stream) {
    if (desc->dtype == F16) {
        relu_cpu_f16(desc, y, x);
        return STATUS_SUCCESS;
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
