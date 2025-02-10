#ifndef BATCH_NORM_H
#define BATCH_NORM_H

#include "../../export.h"
#include "../../operators.h"

typedef struct BatchNormDescriptor {
    Device device;
} BatchNormDescriptor;

typedef BatchNormDescriptor *infiniopBatchNormDescriptor_t;

__C __export infiniopStatus_t infiniopCreateBatchNormDescriptor(infiniopHandle_t handle,
                                                                infiniopBatchNormDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x,
                                                                infiniopTensorDescriptor_t scale,
                                                                infiniopTensorDescriptor_t b,
                                                                infiniopTensorDescriptor_t mean,
                                                                infiniopTensorDescriptor_t var,
                                                                double eps);

__C __export infiniopStatus_t infiniopBatchNorm(infiniopBatchNormDescriptor_t desc,
                                                void *y,
                                                void const *x,
                                                void const *scale,
                                                void const *b,
                                                void const *mean,
                                                void const *var,
                                                void *stream);

__C __export infiniopStatus_t infiniopDestroyBatchNormDescriptor(infiniopBatchNormDescriptor_t desc);

#endif
