#ifndef __CNNL_ROTARY_EMBEDDING_H__
#define __CNNL_ROTARY_EMBEDDING_H__

#include "../../../devices/bang/bang_handle.h"
#include "../../utils.h"
#include "cnnl.h"
#include "cnnl_extra.h"
#include "operators.h"

struct RoPECnnlDescriptor {
    Device device;
    int device_id;
    std::shared_ptr<Pool<cnnlHandle_t>> pool;
    DT dtype;
    cnnlRotaryEmbeddingDescriptor_t ropeDesc;
    cnnlTensorDescriptor_t inDesc;
    cnnlTensorDescriptor_t posDesc;
    cnnlTensorDescriptor_t sinFullDesc;
    cnnlTensorDescriptor_t sinSelectedDesc;
    cnnlTensorDescriptor_t sinSelectedFP16Desc;
};


typedef struct RoPECnnlDescriptor *RoPECnnlDescriptor_t;

infiniopStatus_t cnnlCreateRoPEDescriptor(BangHandle_t handle,
                                          RoPECnnlDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t t,
                                          infiniopTensorDescriptor_t pos_ids,
                                          infiniopTensorDescriptor_t sin_table,
                                          infiniopTensorDescriptor_t cos_table);

infiniopStatus_t cnnlGetRoPEWorkspaceSize(RoPECnnlDescriptor_t desc, uint64_t *size);

infiniopStatus_t cnnlRoPE(RoPECnnlDescriptor_t desc,
                          void *workspace,
                          uint64_t workspace_size,
                          void *t,
                          void const *pos_ids,
                          void const *sin_table,
                          void const *cos_table,
                          void *stream);

infiniopStatus_t cnnlDestroyRoPEDescriptor(RoPECnnlDescriptor_t desc);


#endif// __CNNL_RMS_NORM_H__
