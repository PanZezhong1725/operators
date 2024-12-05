#ifndef RECIPROCAL_H
#define RECIPROCAL_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReciprocalDescriptor {
    Device device;
} ReciprocalDescriptor;

typedef ReciprocalDescriptor *infiniopReciprocalDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReciprocalDescriptor(infiniopHandle_t handle,
                                                                 infiniopReciprocalDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t y,
                                                                 infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopReciprocal(infiniopReciprocalDescriptor_t desc,
                                                 void *y,
                                                 void const *x,
                                                 void *stream);

__C __export infiniopStatus_t infiniopDestroyReciprocalDescriptor(infiniopReciprocalDescriptor_t desc);

#endif
