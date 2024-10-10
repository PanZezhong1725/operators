#ifndef ATTENTION_H
#define ATTENTION_H

#include "../../export.h"
#include "../../operators.h"
#include "../matmul/matmul.h"
#include "../swiglu/swiglu.h"

typedef struct AttentionDescriptor {
    Device device;
} AttentionDescriptor;

typedef AttentionDescriptor *infiniopAttentionDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAttentionDescriptor(infiniopHandle_t handle,
                                                                infiniopAttentionDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t out_desc,
                                                                infiniopTensorDescriptor_t q_desc,
                                                                infiniopTensorDescriptor_t k_desc,
                                                                infiniopTensorDescriptor_t v_desc,
                                                                infiniopTensorDescriptor_t k_cache_desc,
                                                                infiniopTensorDescriptor_t v_cache_desc,
                                                                uint64_t pos);

__C __export infiniopStatus_t infiniopGetAttentionWorkspaceSize(infiniopAttentionDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopAttention(infiniopAttentionDescriptor_t desc,
                                                void *workspace,
                                                uint64_t workspace_size,
                                                void *out,
                                                void *q,
                                                void *k,
                                                void *v,
                                                void *k_cache,
                                                void *v_cache,
                                                void *stream);

__C __export infiniopStatus_t infiniopDestroyAttentionDescriptor(infiniopAttentionDescriptor_t desc);
#endif
