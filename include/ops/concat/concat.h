#ifndef CONCAT_H
#define CONCAT_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ConcatDescriptor {
    Device device;  
} ConcatDescriptor;

typedef ConcatDescriptor *infiniopConcatDescriptor_t;

__C __export infiniopStatus_t infiniopCreateConcatDescriptor(infiniopHandle_t handle,
                                                             infiniopConcatDescriptor_t *desc_ptr,
                                                             infiniopTensorDescriptor_t y,
                                                             infiniopTensorDescriptor_t *x,
                                                             uint64_t num_inputs,
                                                             int64_t axis);

__C __export infiniopStatus_t infiniopConcat(infiniopConcatDescriptor_t desc,
                                             void *y,
                                             void const **x,
                                             void *stream);
                                             
__C __export infiniopStatus_t infiniopDestroyConcatDescriptor(infiniopConcatDescriptor_t desc);

#endif
