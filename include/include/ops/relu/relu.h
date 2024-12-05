#ifndef RELU_H
#define RELU_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReluDescriptor {
    Device device;
} ReluDescriptor;

typedef ReluDescriptor *infiniopReluDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReluDescriptor(infiniopHandle_t handle,
                                                           infiniopReluDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopRelu(infiniopReluDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroyReluDescriptor(infiniopReluDescriptor_t desc);

#endif
