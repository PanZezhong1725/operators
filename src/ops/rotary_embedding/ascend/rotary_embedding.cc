#include "rotary_embedding.h"
#include "../../../utils.h"

extern "C" void rope_kernel_do(void *t, void *pos, void *sin, void *cos,
                               int32_t nt, int32_t nh, int32_t dh,
                               int dtype, void *stream);

infiniopStatus_t ascendCreateRoPEDescriptor(AscendHandle_t handle,
                                            RoPEAscendDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t t,
                                            infiniopTensorDescriptor_t pos_ids,
                                            infiniopTensorDescriptor_t sin_table,
                                            infiniopTensorDescriptor_t cos_table) {
    if (t->ndim != 3 ||
        pos_ids->ndim != 1 ||
        sin_table->ndim != 2 ||
        cos_table->ndim != 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    auto seq_len = t->shape[0];
    auto nh = t->shape[1];
    auto dim = t->shape[2];
    auto total_seq_len = sin_table->shape[0];


    if (dim % 2 != 0) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    if (pos_ids->shape[0] != seq_len ||
        sin_table->shape[1] != dim ||
        cos_table->shape[1] != dim ||
        sin_table->shape[0] != cos_table->shape[0]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    if (t->strides[2] != 1 ||
        pos_ids->strides[0] != 1 ||
        sin_table->strides[1] != 1 ||
        cos_table->strides[1] != 1) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    aclDataType dt;
    if (dtype_eq(t->dt, F16)) {
        dt = aclDataType::ACL_FLOAT16;
    } else if (dtype_eq(t->dt, F32)) {
        dt = aclDataType::ACL_FLOAT;
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    // printf("%lu, %lu, %lu, %lu\n", seq_len, nh, dim, total_seq_len);

    if (!dtype_eq(sin_table->dt, F32) || !dtype_eq(cos_table->dt, F32))
        return STATUS_BAD_TENSOR_DTYPE;

    // if (!dtype_eq(pos_ids->dt, U64))
    //     return STATUS_BAD_TENSOR_DTYPE;

    *desc_ptr = new RoPEAscendDescriptor{
        handle->device,
        handle->device_id,
        dt,
        seq_len,
        nh,
        dim,
        total_seq_len};

    return STATUS_SUCCESS;
}

infiniopStatus_t ascendGetRoPEWorkspaceSize(RoPEAscendDescriptor_t desc,
                                            uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t ascendRoPE(RoPEAscendDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *t,
                            void const *pos_ids,
                            void const *sin_table,
                            void const *cos_table,
                            void *stream) {
    auto nt = static_cast<int>(desc->seq_len);
    auto nh = static_cast<int>(desc->nhead);
    auto dh = static_cast<int>(desc->dim);

    rope_kernel_do(t, (void *) pos_ids, (void *) sin_table, (void *) cos_table,
                   nt, nh, dh, desc->dt, stream);

    return STATUS_SUCCESS;
}

infiniopStatus_t ascendDestroyRoPEDescriptor(RoPEAscendDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
