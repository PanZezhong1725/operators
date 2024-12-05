#ifndef SQRT_H
#define SQRT_H

#include "../../export.h"
#include "../../operators.h"

typedef struct SqrtDescriptor {
    Device device;
} SqrtDescriptor;

typedef SqrtDescriptor *infiniopSqrtDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSqrtDescriptor(infiniopHandle_t handle,
                                                           infiniopSqrtDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopSqrt(infiniopSqrtDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroySqrtDescriptor(infiniopSqrtDescriptor_t desc);

#endif
