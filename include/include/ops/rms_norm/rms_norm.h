#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "../../export.h"
#include "../../operators.h"

typedef struct RMSNormDescriptor {
    Device device;
} RMSNormDescriptor;

typedef RMSNormDescriptor *infiniopRMSNormDescriptor_t;

__C __export infiniopStatus_t infiniopCreateRMSNormDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon);

__C __export infiniopStatus_t infiniopGetRMSNormWorkspaceSize(infiniopRMSNormDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopRMSNorm(infiniopRMSNormDescriptor_t desc, void *workspace, uint64_t workspace_size,
                                              void *y, void const *x, void const *w, void *stream);

__C __export infiniopStatus_t infiniopDestroyRMSNormDescriptor(infiniopRMSNormDescriptor_t desc);

#endif
