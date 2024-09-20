#ifndef MLP_H
#define MLP_H

#include "../../export.h"
#include "../../operators.h"
#include "../matmul/matmul.h"
#include "../swiglu/swiglu.h"

typedef struct MLPDescriptor {
    Device device;
} MLPDescriptor;

typedef MLPDescriptor *infiniopMLPDescriptor_t;

__C __export infiniopStatus_t infiniopCreateMLPDescriptor(infiniopHandle_t handle,
                                                          infiniopMLPDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y_desc,
                                                          infiniopTensorDescriptor_t x_desc,
                                                          infiniopTensorDescriptor_t w12_desc,
                                                          infiniopTensorDescriptor_t w3_desc);

__C __export infiniopStatus_t infiniopGetMLPWorkspaceSize(infiniopMLPDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopMLP(infiniopMLPDescriptor_t desc,
                                          void *workspace,
                                          uint64_t workspace_size,
                                          void *y,
                                          void *x,
                                          void *w12,
                                          void *w3,
                                          float alpha,
                                          bool residual,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyMLPDescriptor(infiniopMLPDescriptor_t desc);
#endif
