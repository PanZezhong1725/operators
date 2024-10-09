#include "rotary_embedding_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../utils.h"
#include <cmath>

struct RoPECpuDescriptor {
    Device device;
    DT dtype;
    uint64_t seq_len;
    uint64_t nhead;
    uint64_t dim;
    uint64_t total_seq_len;
    int64_t strides[2];
};

void rotary_embedding_cpu_f16(RoPECpuDescriptor_t desc,
                              void *t,
                              uint64_t const *pos_ids,
                              float const *sin_table,
                              float const *cos_table) {
    auto nt = desc->seq_len,
         nh = desc->nhead,
         dim = desc->dim,
         dk = dim / 2;

    auto stride_0 = desc->strides[0];
    auto stride_1 = desc->strides[1];

    for (int i = 0; i < nt; ++i) {
        auto sin_ = sin_table + pos_ids[i] * dim;
        auto cos_ = cos_table + pos_ids[i] * dim;
        for (int j = 0; j < nh; ++j) {
            auto t_ = reinterpret_cast<uint16_t *>(t) + i * stride_0 + j * stride_1;
            for (int k = 0; k < dk; ++k) {
                auto a = f16_to_f32(t_[2 * k]);
                auto b = f16_to_f32(t_[2 * k + 1]);
                float sin0 = sin_[k * 2], cos0 = cos_[k * 2];
                float sin1 = sin_[k * 2 + 1], cos1 = cos_[k * 2 + 1];
                t_[2 * k] = f32_to_f16(a * cos0 - b * sin0);
                t_[2 * k + 1] = f32_to_f16(a * sin1 + b * cos1);
            }
        }
    }
}


infiniopStatus_t cpuCreateRoPEDescriptor(CpuHandle_t handle,
                                         RoPECpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t t,
                                         infiniopTensorDescriptor_t pos_ids,
                                         infiniopTensorDescriptor_t sin_table,
                                         infiniopTensorDescriptor_t cos_table) {

    if (desc_ptr == nullptr)
        return STATUS_MEMORY_NOT_ALLOCATED;

    if (t->ndim != 3 ||
        pos_ids->ndim != 1 ||
        sin_table->ndim != 2 ||
        cos_table->ndim != 2)
        return STATUS_BAD_TENSOR_SHAPE;

    auto seq_len = t->shape[0];
    auto nhead = t->shape[1];
    auto dim = t->shape[2];
    auto total_seq_len = sin_table->shape[0];

    if (dim % 2 != 0)
        return STATUS_BAD_TENSOR_SHAPE;

    if (pos_ids->shape[0] != seq_len ||
        sin_table->shape[1] != dim ||
        cos_table->shape[1] != dim ||
        sin_table->shape[0] != cos_table->shape[0])
        return STATUS_BAD_TENSOR_SHAPE;

    if (t->strides[2] != 1 ||
        pos_ids->strides[0] != 1 ||
        sin_table->strides[1] != 1 ||
        cos_table->strides[1] != 1)
        return STATUS_BAD_TENSOR_STRIDES;

    if (!dtype_eq(t->dt, F16))
        return STATUS_BAD_TENSOR_DTYPE;

    if (!dtype_eq(sin_table->dt, F32) || !dtype_eq(cos_table->dt, F32))
        return STATUS_BAD_TENSOR_DTYPE;

    // if (!dtype_eq(pos_ids->dt, U64))
    //     return STATUS_BAD_TENSOR_DTYPE;

    *desc_ptr = new RoPECpuDescriptor{
        handle->device,
        t->dt,
        seq_len,
        nhead,
        dim,
        total_seq_len,
        {t->strides[0], t->strides[1]}};

    return STATUS_SUCCESS;
}


infiniopStatus_t cpuGetRoPEWorkspaceSize(RoPECpuDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}


infiniopStatus_t cpuRoPE(RoPECpuDescriptor_t desc,
                         void *workspace,
                         uint64_t workspace_size,
                         void *t,
                         void const *pos_ids,
                         void const *sin_table,
                         void const *cos_table,
                         void *stream) {
    if (t == nullptr || pos_ids == nullptr || sin_table == nullptr || cos_table == nullptr)
        return STATUS_BAD_PARAM;

    if (dtype_eq(desc->dtype, F16)) {
        rotary_embedding_cpu_f16(desc, t,
                                 reinterpret_cast<uint64_t const *>(pos_ids),
                                 reinterpret_cast<float const *>(sin_table),
                                 reinterpret_cast<float const *>(cos_table));
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    return STATUS_SUCCESS;
}


infiniopStatus_t cpuDestroyRoPEDescriptor(RoPECpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
