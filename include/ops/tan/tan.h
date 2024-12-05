#ifndef TAN_H
#define TAN_H

#include "../../export.h"
#include "../../operators.h"

typedef struct TanDescriptor {
    Device device;
} TanDescriptor;

typedef TanDescriptor *infiniopTanDescriptor_t;

__C __export infiniopStatus_t infiniopCreateTanDescriptor(infiniopHandle_t handle,
                                                          infiniopTanDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y,
                                                          infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopTan(infiniopTanDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyTanDescriptor(infiniopTanDescriptor_t desc);

#endif
