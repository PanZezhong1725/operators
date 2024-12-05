#ifndef COSH_H
#define COSH_H

#include "../../export.h"
#include "../../operators.h"

typedef struct CoshDescriptor {
    Device device;
} CoshDescriptor;

typedef CoshDescriptor *infiniopCoshDescriptor_t;

__C __export infiniopStatus_t infiniopCreateCoshDescriptor(infiniopHandle_t handle,
                                                           infiniopCoshDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopCosh(infiniopCoshDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroyCoshDescriptor(infiniopCoshDescriptor_t desc);

#endif
