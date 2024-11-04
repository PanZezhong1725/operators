#ifndef POOLING_H
#define POOLING_H

#include "../../export.h"
#include "../../operators.h"

typedef struct PoolingDescriptor {
    Device device;
} PoolingDescriptor;
typedef PoolingDescriptor *infiniopPoolingDescriptor_t;

__C __export infiniopStatus_t infiniopCreatePoolingDescriptor(infiniopHandle_t handle,
                                                              infiniopPoolingDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t y,
                                                              infiniopTensorDescriptor_t x,
                                                              void const *kernel_shape,
                                                              void const *pads,
                                                              void const *strides,
                                                              uint64_t n,
                                                              int pooling_type);

__C __export infiniopStatus_t infiniopPooling(infiniopPoolingDescriptor_t desc, void *y, void const *x, void *stream);

__C __export infiniopStatus_t infiniopDestroyPoolingDescriptor(infiniopPoolingDescriptor_t desc);
#endif
