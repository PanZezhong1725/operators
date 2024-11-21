#ifndef MAX_H
#define MAX_H

#include "../../export.h"
#include "../../operators.h"

typedef struct MaxDescriptor {
    Device device;
} MaxDescriptor;

typedef MaxDescriptor *infiniopMaxDescriptor_t;

__C __export infiniopStatus_t infiniopCreateMaxDescriptor(infiniopHandle_t handle,
                                                          infiniopMaxDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c,
                                                          infiniopTensorDescriptor_t a,
                                                          infiniopTensorDescriptor_t b);

__C __export infiniopStatus_t infiniopMax(infiniopMaxDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyMaxDescriptor(infiniopMaxDescriptor_t desc);

#endif
