#ifndef SIN_H
#define SIN_H

#include "../../export.h"
#include "../../operators.h"

typedef struct SinDescriptor {
    Device device;
} SinDescriptor;

typedef SinDescriptor *infiniopSinDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSinDescriptor(infiniopHandle_t handle,
                                                          infiniopSinDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y,
                                                          infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopSin(infiniopSinDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroySinDescriptor(infiniopSinDescriptor_t desc);

#endif
