#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include "../../export.h"
#include "../../operators.h"

typedef struct TransposeDescriptor {
    Device device;
} TransposeDescriptor;

typedef TransposeDescriptor *infiniopTransposeDescriptor_t;

__C __export infiniopStatus_t
infiniopCreateTransposeDescriptor(infiniopHandle_t handle,
                                  infiniopTransposeDescriptor_t *desc_ptr,
                                  infiniopTensorDescriptor_t y,
                                  infiniopTensorDescriptor_t x,
                                  uint64_t const *perm,
                                  uint64_t n);

__C __export infiniopStatus_t infiniopTranspose(infiniopTransposeDescriptor_t desc,
                                                void *y,
                                                void const *x,
                                                void *stream);

__C __export infiniopStatus_t infiniopDestroyTransposeDescriptor(infiniopTransposeDescriptor_t desc);

#endif
