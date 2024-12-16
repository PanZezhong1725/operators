#include "rotary_embedding.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateRoPEDescriptor(CudaHandle_t handle,
                                          RoPECudaDescriptor_t *desc_ptr,
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

    // TODO: support larger dim in the future
    if (dim / 2 > MAX_THREADS_PER_BLOCK) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

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

    *desc_ptr = new RoPECudaDescriptor{
        handle->device,
        handle->device_id,
        t->dt,
        seq_len,
        nhead,
        dim,
        total_seq_len,
        {t->strides[0], t->strides[1]}};

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGetRoPEWorkspaceSize(RoPECudaDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}


infiniopStatus_t cudaDestroyRoPEDescriptor(RoPECudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
