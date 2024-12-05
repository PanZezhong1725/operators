#ifndef ASIN_H
#define ASIN_H

#include "../../export.h"
#include "../../operators.h"

typedef struct AsinDescriptor {
    Device device;
} AsinDescriptor;

typedef AsinDescriptor *infiniopAsinDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAsinDescriptor(infiniopHandle_t handle,
                                                           infiniopAsinDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopAsin(infiniopAsinDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroyAsinDescriptor(infiniopAsinDescriptor_t desc);

#endif
