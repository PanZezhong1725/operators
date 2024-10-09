#ifndef MATMUL_H
#define MATMUL_H

#include "../../export.h"
#include "../../operators.h"

typedef struct MatmulDescriptor {
    Device device;
} MatmulDescriptor;

typedef MatmulDescriptor *infiniopMatmulDescriptor_t;

__C __export infiniopStatus_t infiniopCreateMatmulDescriptor(infiniopHandle_t handle,
                                                             infiniopMatmulDescriptor_t *desc_ptr,
                                                             infiniopTensorDescriptor_t c_desc,
                                                             float alpha,
                                                             infiniopTensorDescriptor_t a_desc,
                                                             infiniopTensorDescriptor_t b_desc,
                                                             float beta);

__C __export infiniopStatus_t infiniopGetMatmulWorkspaceSize(infiniopMatmulDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopMatmul(infiniopMatmulDescriptor_t desc,
                                             void *workspace,
                                             uint64_t workspace_size,
                                             void *c,
                                             void const *a,
                                             void const *b,
                                             void *stream);

__C __export infiniopStatus_t infiniopDestroyMatmulDescriptor(infiniopMatmulDescriptor_t desc);

#endif
