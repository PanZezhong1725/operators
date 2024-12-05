#ifndef TANH_H
#define TANH_H

#include "../../export.h"
#include "../../operators.h"

typedef struct TanhDescriptor {
    Device device;
} TanhDescriptor;

typedef TanhDescriptor *infiniopTanhDescriptor_t;

__C __export infiniopStatus_t infiniopCreateTanhDescriptor(infiniopHandle_t handle,
                                                           infiniopTanhDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopTanh(infiniopTanhDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroyTanhDescriptor(infiniopTanhDescriptor_t desc);

#endif
