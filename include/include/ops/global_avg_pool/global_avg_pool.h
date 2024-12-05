#ifndef GLOBAL_AVG_POOL_H
#define GLOBAL_AVG_POOL_H

#include "../../export.h"
#include "../../operators.h"

typedef struct GlobalAvgPoolDescriptor {
    Device device;
} GlobalAvgPoolDescriptor;

typedef GlobalAvgPoolDescriptor *infiniopGlobalAvgPoolDescriptor_t;

__C __export infiniopStatus_t infiniopCreateGlobalAvgPoolDescriptor(infiniopHandle_t handle,
                                                                    infiniopGlobalAvgPoolDescriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t y,
                                                                    infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopGetGlobalAvgPoolWorkspaceSize(infiniopGlobalAvgPoolDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopGlobalAvgPool(infiniopGlobalAvgPoolDescriptor_t desc,
                                                    void *workspace, uint64_t workspace_size,
                                                    void *y, void const *x, void *stream);

__C __export infiniopStatus_t infiniopDestroyGlobalAvgPoolDescriptor(infiniopGlobalAvgPoolDescriptor_t desc);

#endif
