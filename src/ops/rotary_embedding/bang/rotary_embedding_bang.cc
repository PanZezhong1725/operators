#include "rotary_embedding_bang.h"
#include "../../utils.h"


infiniopStatus_t bangCreateRoPEDescriptor(BangHandle_t handle,
                                          RoPEBangDescriptor_t *desc_ptr,
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

    if (!dtype_eq(pos_ids->dt, U64))
        return STATUS_BAD_TENSOR_DTYPE;
    int stride_0 = static_cast<int>(t->strides[0]);
    int stride_1 = static_cast<int>(t->strides[1]);
    *desc_ptr = new RoPEBangDescriptor{
        handle->device,
        handle->device_id,
        t->dt,
        seq_len,
        nhead,
        dim,
        total_seq_len,
        stride_0, stride_1};

    return STATUS_SUCCESS;
}


infiniopStatus_t bangGetRoPEWorkspaceSize(RoPEBangDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}


infiniopStatus_t bangDestroyRoPEDescriptor(RoPEBangDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
