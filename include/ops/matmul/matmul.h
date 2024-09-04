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
                                                             infiniopTensorDescriptor_t a_desc,
                                                             infiniopTensorDescriptor_t b_desc);

__C __export infiniopStatus_t infiniopGetMatmulWorkspaceSize(infiniopMatmulDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopMatmul(infiniopMatmulDescriptor_t desc,
                                             void *workspace,
                                             uint64_t workspace_size,
                                             void *c,
                                             void *a,
                                             void *b,
                                             float alpha,
                                             float beta,
                                             void *stream);

__C __export infiniopStatus_t infiniopDestroyMatmulDescriptor(infiniopMatmulDescriptor_t desc);

#endif
