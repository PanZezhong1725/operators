#ifndef EXPAND_H
#define EXPAND_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ExpandDescriptor {
    Device device;
} ExpandDescriptor;

typedef ExpandDescriptor *infiniopExpandDescriptor_t;

__C __export infiniopStatus_t infiniopCreateExpandDescriptor(infiniopHandle_t handle,
                                                             infiniopExpandDescriptor_t *desc_ptr,
                                                             infiniopTensorDescriptor_t y,
                                                             infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopExpand(infiniopExpandDescriptor_t desc,
                                             void *y,
                                             void const *x,
                                             void *stream);

__C __export infiniopStatus_t infiniopDestroyExpandDescriptor(infiniopExpandDescriptor_t desc);

#endif
