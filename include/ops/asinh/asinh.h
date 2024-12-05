#ifndef ASINH_H
#define ASINH_H

#include "../../export.h"
#include "../../operators.h"

typedef struct AsinhDescriptor {
    Device device;
} AsinhDescriptor;

typedef AsinhDescriptor *infiniopAsinhDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAsinhDescriptor(infiniopHandle_t handle,
                                                            infiniopAsinhDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y,
                                                            infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopAsinh(infiniopAsinhDescriptor_t desc,
                                            void *y,
                                            void const *x,
                                            void *stream);

__C __export infiniopStatus_t infiniopDestroyAsinhDescriptor(infiniopAsinhDescriptor_t desc);

#endif
