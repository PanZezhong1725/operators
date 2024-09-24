#ifndef __BANG_ROTARY_EMBEDDING_H__
#define __BANG_ROTARY_EMBEDDING_H__

#include "../../../devices/bang/bang_handle.h"
#include "../../utils.h"
#include "operators.h"

struct RoPEBangDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t seq_len;
    uint64_t nhead;
    uint64_t dim;
    uint64_t total_seq_len;
    int stride_0;
    int stride_1;
};


typedef struct RoPEBangDescriptor *RoPEBangDescriptor_t;

infiniopStatus_t bangCreateRoPEDescriptor(BangHandle_t handle,
                                          RoPEBangDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t t,
                                          infiniopTensorDescriptor_t pos_ids,
                                          infiniopTensorDescriptor_t sin_table,
                                          infiniopTensorDescriptor_t cos_table);

infiniopStatus_t bangGetRoPEWorkspaceSize(RoPEBangDescriptor_t desc, uint64_t *size);

infiniopStatus_t bangRoPE(RoPEBangDescriptor_t desc,
                          void *workspace,
                          uint64_t workspace_size,
                          void *t,
                          void const *pos_ids,
                          void const *sin_table,
                          void const *cos_table,
                          void *stream);

infiniopStatus_t bangDestroyRoPEDescriptor(RoPEBangDescriptor_t desc);


#endif// __BANG_RMS_NORM_H__
