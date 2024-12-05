#ifndef NEG_H
#define NEG_H

#include "../../export.h"
#include "../../operators.h"

typedef struct NegDescriptor {
    Device device;
} NegDescriptor;

typedef NegDescriptor *infiniopNegDescriptor_t;

__C __export infiniopStatus_t infiniopCreateNegDescriptor(infiniopHandle_t handle,
                                                          infiniopNegDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y,
                                                          infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopNeg(infiniopNegDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyNegDescriptor(infiniopNegDescriptor_t desc);

#endif
