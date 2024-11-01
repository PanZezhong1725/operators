#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../../export.h"
#include "../../operators.h"

typedef struct SoftmaxDescriptor {
    Device device;
} SoftmaxDescriptor;

typedef SoftmaxDescriptor *infiniopSoftmaxDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSoftmaxDescriptor(infiniopHandle_t handle,
                                                              infiniopSoftmaxDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t output_desc);


__C __export infiniopStatus_t infiniopSoftmax(infiniopSoftmaxDescriptor_t desc,
                                              void const *input,
                                              int axis,
                                              void *output,
                                              void *stream);

__C __export infiniopStatus_t infiniopDestroySoftmaxDescriptor(infiniopSoftmaxDescriptor_t desc);


#endif
