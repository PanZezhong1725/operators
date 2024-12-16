#ifndef __NV_GPU_ROTARY_EMBEDDING_H__
#define __NV_GPU_ROTARY_EMBEDDING_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"

struct RoPECudaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t seq_len;
    uint64_t nhead;
    uint64_t dim;
    uint64_t total_seq_len;
    int64_t strides[2];
};

typedef struct RoPECudaDescriptor *RoPECudaDescriptor_t;

infiniopStatus_t cudaCreateRoPEDescriptor(CudaHandle_t handle,
                                          RoPECudaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t t,
                                          infiniopTensorDescriptor_t pos_ids,
                                          infiniopTensorDescriptor_t sin_table,
                                          infiniopTensorDescriptor_t cos_table);

infiniopStatus_t cudaGetRoPEWorkspaceSize(RoPECudaDescriptor_t desc, uint64_t *size);

infiniopStatus_t cudaRoPE(RoPECudaDescriptor_t desc,
                          void *workspace,
                          unsigned long int workspace_size,
                          void *t,
                          void const *pos_ids,
                          void const *sin_table,
                          void const *cos_table,
                          void *stream);

infiniopStatus_t cudaDestroyRoPEDescriptor(RoPECudaDescriptor_t desc);

#endif// __NV_GPU_ROTARY_EMBEDDING_H__
