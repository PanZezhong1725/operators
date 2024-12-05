#ifndef ATAN_H
#define ATAN_H

#include "../../export.h"
#include "../../operators.h"

typedef struct AtanDescriptor {
    Device device;
} AtanDescriptor;

typedef AtanDescriptor *infiniopAtanDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAtanDescriptor(infiniopHandle_t handle,
                                                           infiniopAtanDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopAtan(infiniopAtanDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroyAtanDescriptor(infiniopAtanDescriptor_t desc);

#endif
