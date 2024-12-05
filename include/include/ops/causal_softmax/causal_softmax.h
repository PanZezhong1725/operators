#ifndef CAUSAL_SOFTMAX_H
#define CAUSAL_SOFTMAX_H

#include "../../export.h"
#include "../../operators.h"

typedef struct CausalSoftmaxDescriptor {
    Device device;
} CausalSoftmaxDescriptor;

typedef CausalSoftmaxDescriptor *infiniopCausalSoftmaxDescriptor_t;

__C __export infiniopStatus_t infiniopCreateCausalSoftmaxDescriptor(infiniopHandle_t handle,
                                                                    infiniopCausalSoftmaxDescriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t y_desc);

__C __export infiniopStatus_t infiniopGetCausalSoftmaxWorkspaceSize(infiniopCausalSoftmaxDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopCausalSoftmax(infiniopCausalSoftmaxDescriptor_t desc,
                                                    void *workspace,
                                                    uint64_t workspace_size,
                                                    void *data,
                                                    void *stream);

__C __export infiniopStatus_t infiniopDestroyCausalSoftmaxDescriptor(infiniopCausalSoftmaxDescriptor_t desc);


#endif
