#ifndef ABS_H
#define ABS_H

#include "../../export.h"
#include "../../operators.h"

typedef struct AbsDescriptor {
    Device device;
} AbsDescriptor;

typedef AbsDescriptor *infiniopAbsDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAbsDescriptor(infiniopHandle_t handle,
                                                          infiniopAbsDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y,
                                                          infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopAbs(infiniopAbsDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyAbsDescriptor(infiniopAbsDescriptor_t desc);

#endif
