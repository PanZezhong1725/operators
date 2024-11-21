#ifndef MIN_H
#define MIN_H

#include "../../export.h"
#include "../../operators.h"

typedef struct MinDescriptor {
    Device device;
} MinDescriptor;

typedef MinDescriptor *infiniopMinDescriptor_t;

__C __export infiniopStatus_t infiniopCreateMinDescriptor(infiniopHandle_t handle,
                                                          infiniopMinDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c,
                                                          infiniopTensorDescriptor_t a,
                                                          infiniopTensorDescriptor_t b);

__C __export infiniopStatus_t infiniopMin(infiniopMinDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyMinDescriptor(infiniopMinDescriptor_t desc);

#endif
