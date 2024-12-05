#ifndef SIGMOID_H
#define SIGMOID_H

#include "../../export.h"
#include "../../operators.h"

typedef struct SigmoidDescriptor {
    Device device;
} SigmoidDescriptor;

typedef SigmoidDescriptor *infiniopSigmoidDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSigmoidDescriptor(infiniopHandle_t handle,
                                                              infiniopSigmoidDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t y,
                                                              infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopSigmoid(infiniopSigmoidDescriptor_t desc,
                                              void *y,
                                              void const *x,
                                              void *stream);

__C __export infiniopStatus_t infiniopDestroySigmoidDescriptor(infiniopSigmoidDescriptor_t desc);

#endif
