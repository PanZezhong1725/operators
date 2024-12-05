#ifndef CEIL_H
#define CEIL_H

#include "../../export.h"
#include "../../operators.h"

typedef struct CeilDescriptor {
    Device device;
} CeilDescriptor;

typedef CeilDescriptor *infiniopCeilDescriptor_t;

__C __export infiniopStatus_t infiniopCreateCeilDescriptor(infiniopHandle_t handle,
                                                           infiniopCeilDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopCeil(infiniopCeilDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroyCeilDescriptor(infiniopCeilDescriptor_t desc);

#endif
