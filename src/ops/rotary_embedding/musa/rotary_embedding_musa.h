#ifndef __MUSA_ROTARY_EMBEDDING_H__
#define __MUSA_ROTARY_EMBEDDING_H__

#include "../../../devices/musa/musa_handle.h"
#include "operators.h"

struct RoPEMusaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t seq_len;
    uint64_t nhead;
    uint64_t dim;
    uint64_t total_seq_len;
    int64_t strides[2];
};

typedef struct RoPEMusaDescriptor *RoPEMusaDescriptor_t;

infiniopStatus_t musaCreateRoPEDescriptor(MusaHandle_t handle,
                                          RoPEMusaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t t,
                                          infiniopTensorDescriptor_t pos_ids,
                                          infiniopTensorDescriptor_t sin_table,
                                          infiniopTensorDescriptor_t cos_table);

infiniopStatus_t musaGetRoPEWorkspaceSize(RoPEMusaDescriptor_t desc, unsigned long int *size);

infiniopStatus_t musaRoPE(RoPEMusaDescriptor_t desc,
                          void *workspace,
                          unsigned long int workspace_size,
                          void *t,
                          void const *pos_ids,
                          void const *sin_table,
                          void const *cos_table,
                          void *stream);

infiniopStatus_t musaDestroyRoPEDescriptor(RoPEMusaDescriptor_t desc);

#endif// __NV_GPU_ROTARY_EMBEDDING_H__
