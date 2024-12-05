#ifndef ROTARY_EMBEDDING_H
#define ROTARY_EMBEDDING_H

#include "../../export.h"
#include "../../operators.h"

typedef struct RoPEDescriptor RoPEDescriptor;
typedef RoPEDescriptor *infiniopRoPEDescriptor_t;

__C __export infiniopStatus_t infiniopCreateRoPEDescriptor(
    infiniopHandle_t handle,
    infiniopRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t t,
    infiniopTensorDescriptor_t pos_ids,
    infiniopTensorDescriptor_t sin_table,
    infiniopTensorDescriptor_t cos_table);

__C __export infiniopStatus_t infiniopGetRoPEWorkspaceSize(infiniopRoPEDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopRoPE(
    infiniopRoPEDescriptor_t desc,
    void *workspace,
    uint64_t workspace_size,
    void *t,
    void const *pos_ids,
    void const *sin_table,
    void const *cos_table,
    void *stream);

__C __export infiniopStatus_t infiniopDestroyRoPEDescriptor(infiniopRoPEDescriptor_t desc);

#endif
