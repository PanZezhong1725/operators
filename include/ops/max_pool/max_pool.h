#ifndef MAX_POOL_H
#define MAX_POOL_H

#include "../../export.h"
#include "../../operators.h"

typedef struct MaxPoolDescriptor {
    Device device;
} MaxPoolDescriptor;
typedef MaxPoolDescriptor *infiniopMaxPoolDescriptor_t;

__C __export infiniopStatus_t infiniopCreateMaxPoolDescriptor(infiniopHandle_t handle,
                                                              infiniopMaxPoolDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t y,
                                                              infiniopTensorDescriptor_t x,
                                                              uint64_t const *kernel_shape,
                                                              uint64_t const *pads,
                                                              int64_t const *strides,
                                                              uint64_t n);

__C __export infiniopStatus_t infiniopGetMaxPoolWorkspaceSize(infiniopMaxPoolDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopMaxPool(infiniopMaxPoolDescriptor_t desc,
                                              void *workspace, uint64_t workspace_size,
                                              void *y, void const *x, void *stream);

__C __export infiniopStatus_t infiniopDestroyMaxPoolDescriptor(infiniopMaxPoolDescriptor_t desc);
#endif
