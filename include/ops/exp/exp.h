#ifndef EXP_H
#define EXP_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ExpDescriptor {
    Device device;
} ExpDescriptor;

typedef ExpDescriptor *infiniopExpDescriptor_t;

__C __export infiniopStatus_t infiniopCreateExpDescriptor(infiniopHandle_t handle,
                                                          infiniopExpDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y,
                                                          infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopExp(infiniopExpDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyExpDescriptor(infiniopExpDescriptor_t desc);

#endif
