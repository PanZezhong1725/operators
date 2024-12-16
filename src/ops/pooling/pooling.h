#ifndef POOLING_H
#define POOLING_H

#include "export.h"
#include "operators.h"

typedef struct PoolingDescriptor {
    Device device;
} PoolingDescriptor;
typedef PoolingDescriptor *infiniopPoolingDescriptor_t;

__C infiniopStatus_t infiniopCreatePoolingDescriptor(infiniopHandle_t handle,
                                                     infiniopPoolingDescriptor_t *desc_ptr,
                                                     infiniopTensorDescriptor_t y,
                                                     infiniopTensorDescriptor_t x,
                                                     uint64_t const *kernel_shape,
                                                     uint64_t const *pads,
                                                     int64_t const *strides,
                                                     uint64_t n,
                                                     int pooling_type);

__C infiniopStatus_t infiniopGetPoolingWorkspaceSize(infiniopPoolingDescriptor_t desc, uint64_t *size);

__C infiniopStatus_t infiniopPooling(infiniopPoolingDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void *stream);

__C infiniopStatus_t infiniopDestroyPoolingDescriptor(infiniopPoolingDescriptor_t desc);
#endif
