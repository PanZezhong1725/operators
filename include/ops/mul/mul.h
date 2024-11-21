#ifndef MUL_H
#define MUL_H

#include "../../export.h"
#include "../../operators.h"

typedef struct MulDescriptor {
    Device device;
} MulDescriptor;

typedef MulDescriptor *infiniopMulDescriptor_t;

__C __export infiniopStatus_t infiniopCreateMulDescriptor(infiniopHandle_t handle,
                                                          infiniopMulDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c,
                                                          infiniopTensorDescriptor_t a,
                                                          infiniopTensorDescriptor_t b);

__C __export infiniopStatus_t infiniopMul(infiniopMulDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyMulDescriptor(infiniopMulDescriptor_t desc);

#endif
