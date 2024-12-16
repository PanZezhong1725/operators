#ifndef GEMM_H
#define GEMM_H

#include "../../export.h"
#include "../../operators.h"

typedef struct GEMMDescriptor {
    Device device;
} GEMMDescriptor;

typedef GEMMDescriptor *infiniopGEMMDescriptor_t;

__C __export infiniopStatus_t infiniopCreateGEMMDescriptor(infiniopHandle_t handle,
                                                           infiniopGEMMDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y_desc,
                                                           infiniopTensorDescriptor_t a_desc,
                                                           infiniopTensorDescriptor_t b_desc,
                                                           infiniopTensorDescriptor_t c_desc,
                                                           float alpha,
                                                           float beta,
                                                           char transA,
                                                           char transB);

__C __export infiniopStatus_t infiniopGetGEMMWorkspaceSize(infiniopGEMMDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopGEMM(infiniopGEMMDescriptor_t desc,
                                           void *workspace,
                                           uint64_t workspace_size,
                                           void *y,
                                           void const *a,
                                           void const *b,
                                           void const *c,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroyGEMMDescriptor(infiniopGEMMDescriptor_t desc);
#endif
