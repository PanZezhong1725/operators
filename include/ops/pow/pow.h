#ifndef POW_H
#define POW_H

#include "../../export.h"
#include "../../operators.h"

typedef struct PowDescriptor {
    Device device;
} PowDescriptor;

typedef PowDescriptor *infiniopPowDescriptor_t;

__C __export infiniopStatus_t infiniopCreatePowDescriptor(infiniopHandle_t handle,
                                                          infiniopPowDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c,
                                                          infiniopTensorDescriptor_t a,
                                                          infiniopTensorDescriptor_t b);

__C __export infiniopStatus_t infiniopPow(infiniopPowDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyPowDescriptor(infiniopPowDescriptor_t desc);

#endif
