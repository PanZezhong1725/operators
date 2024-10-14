#include "swiglu_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../utils.h"
#include <cmath>


infiniopStatus_t cpuCreateSwiGLUDescriptor(infiniopHandle_t handle,
                                           SwiGLUCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t c_desc,
                                           infiniopTensorDescriptor_t a_desc,
                                           infiniopTensorDescriptor_t b_desc) {
    if (c_desc->ndim != 2 || a_desc->ndim != 2 || b_desc->ndim != 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    DT dtype = c_desc->dt;

    if (!dtype_eq(dtype, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    if (a_desc->strides[1] != 1 || b_desc->strides[1] != 1 || c_desc->strides[1] != 1) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    uint64_t seq_len = c_desc->shape[0],
             di = c_desc->shape[1];

    uint64_t stride_a = a_desc->strides[0],
             stride_b = b_desc->strides[0],
             stride_c = c_desc->strides[0];


    if (a_desc->shape[0] != seq_len || a_desc->shape[1] != di || !dtype_eq(a_desc->dt, dtype) ||
        b_desc->shape[0] != seq_len || b_desc->shape[1] != di || !dtype_eq(b_desc->dt, dtype)) {
        return STATUS_BAD_PARAM;
    }

    *desc_ptr = new SwiGLUCpuDescriptor{DevCpu,
                                        dtype,
                                        seq_len,
                                        di,
                                        stride_a,
                                        stride_b,
                                        stride_c};
    return STATUS_SUCCESS;
}

inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

void swiglu_cpu_f16(SwiGLUCpuDescriptor_t desc, void *c, void const *a, void const *b) {

    auto seq_len = desc->seq_len,
         di = desc->di;

    auto stride_a = desc->stride_a,
         stride_b = desc->stride_b,
         stride_c = desc->stride_c;

    for (int i = 0; i < seq_len; ++i) {
        auto a_ = reinterpret_cast<const uint16_t *>(a) + i * stride_a;
        auto b_ = reinterpret_cast<const uint16_t *>(b) + i * stride_b;
        auto c_ = reinterpret_cast<uint16_t *>(c) + i * stride_c;
        for (int j = 0; j < di; ++j) {
            auto a__ = f16_to_f32(a_[j]);
            auto b__ = f16_to_f32(b_[j]);

            c_[j] = f32_to_f16(a__ * silu(b__));
        }
    }
}

infiniopStatus_t cpuSwiGLU(SwiGLUCpuDescriptor_t desc,
                           void *c,
                           void const *a,
                           void const *b,
                           void *stream) {
    if (dtype_eq(desc->dtype, F16)) {
        swiglu_cpu_f16(desc, c, a, b);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}

infiniopStatus_t cpuDestroySwiGLUDescriptor(SwiGLUCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
