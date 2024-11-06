#ifndef AVG_POOL_H
#define AVG_POOL_H

#include "../../export.h"
#include "../../operators.h"

typedef struct AvgPoolDescriptor {
    Device device;
} AvgPoolDescriptor;
typedef AvgPoolDescriptor *infiniopAvgPoolDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAvgPoolDescriptor(infiniopHandle_t handle,
                                                              infiniopAvgPoolDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t y,
                                                              infiniopTensorDescriptor_t x,
                                                              uint64_t const *kernel_shape,
                                                              uint64_t const *pads,
                                                              int64_t const *strides,
                                                              uint64_t n);

__C __export infiniopStatus_t infiniopGetAvgPoolWorkspaceSize(infiniopAvgPoolDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopAvgPool(infiniopAvgPoolDescriptor_t desc,
                                              void *workspace, uint64_t workspace_size,
                                              void *y, void const *x, void *stream);

__C __export infiniopStatus_t infiniopDestroyAvgPoolDescriptor(infiniopAvgPoolDescriptor_t desc);
#endif
