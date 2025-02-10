#include "batch_norm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateBatchNormDescriptor(infiniopHandle_t,
                                              BatchNormCpuDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t y,
                                              infiniopTensorDescriptor_t x,
                                              infiniopTensorDescriptor_t scale,
                                              infiniopTensorDescriptor_t b,
                                              infiniopTensorDescriptor_t mean,
                                              infiniopTensorDescriptor_t var,
                                              double eps) {
    uint64_t ndim = y->ndim;
    if (ndim != x->ndim || scale->ndim != b->ndim || scale->ndim != mean->ndim || scale->ndim != var->ndim || scale->ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (y->shape[i] != x->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (x->shape[1] != scale->shape[0] || scale->shape[0] != b->shape[0] || scale->shape[0] != mean->shape[0] || scale->shape[0] != var->shape[0]) {
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
    if (eps < 0) {
        return STATUS_BAD_PARAM;
    }

    uint64_t spatial_data_size = std::accumulate(x->shape + 2, x->shape + x->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t batch_size = x->shape[0];
    uint64_t channel_size = x->shape[1];

    *desc_ptr = new BatchNormCpuDescriptor{
        DevCpu,
        y->dt,
        batch_size,
        channel_size,
        spatial_data_size,
        channel_size * spatial_data_size,
        eps,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyBatchNormDescriptor(BatchNormCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata, typename Pdata>
infiniopStatus_t batch_norm_cpu(BatchNormCpuDescriptor_t desc, void *y, void const *x,
                                void const *scale, void const *b, void const *mean, void const *var) {
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto scale_ = reinterpret_cast<Pdata const *>(scale);
    auto b_ = reinterpret_cast<Pdata const *>(b);
    auto mean_ = reinterpret_cast<Pdata const *>(mean);
    auto var_ = reinterpret_cast<Pdata const *>(var);
    auto y_ = reinterpret_cast<Tdata *>(y);

#pragma omp parallel for collapse(3)
    for (uint64_t i = 0; i < desc->batch_size; ++i) {
        for (uint64_t c = 0; c < desc->channel_size; ++c) {
            for (uint64_t j = 0; j < desc->spatial_data_size; ++j) {
                auto idx = (i * desc->channel_size + c) * desc->spatial_data_size + j;
                Pdata invsqrt = 1 / std::sqrt(var_[c] + desc->eps);
                if constexpr (std::is_same<Tdata, uint16_t>::value) {
                    y_[idx] = f32_to_f16((f16_to_f32(x_[idx]) - mean_[c]) * invsqrt * scale_[c] + b_[c]);
                } else {
                    y_[idx] = (x_[idx] - mean_[c]) * invsqrt * scale_[c] + b_[c];
                }
            }
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuBatchNorm(BatchNormCpuDescriptor_t desc,
                              void *y, void const *x, void const *scale, void const *b,
                              void const *mean, void const *var, void *stream) {
    if (desc->dtype == F16) {
        return batch_norm_cpu<uint16_t, float>(desc, y, x, scale, b, mean, var);
    }
    if (desc->dtype == F32) {
        return batch_norm_cpu<float, float>(desc, y, x, scale, b, mean, var);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
