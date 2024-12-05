#ifndef SIGN_H
#define SIGN_H

#include "../../export.h"
#include "../../operators.h"

typedef struct SignDescriptor {
    Device device;
} SignDescriptor;

typedef SignDescriptor *infiniopSignDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSignDescriptor(infiniopHandle_t handle,
                                                           infiniopSignDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopSign(infiniopSignDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroySignDescriptor(infiniopSignDescriptor_t desc);

#endif
