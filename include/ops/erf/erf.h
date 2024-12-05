#ifndef ERF_H
#define ERF_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ErfDescriptor {
    Device device;
} ErfDescriptor;

typedef ErfDescriptor *infiniopErfDescriptor_t;

__C __export infiniopStatus_t infiniopCreateErfDescriptor(infiniopHandle_t handle,
                                                          infiniopErfDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y,
                                                          infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopErf(infiniopErfDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyErfDescriptor(infiniopErfDescriptor_t desc);

#endif
