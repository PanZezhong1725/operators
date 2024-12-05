#ifndef ACOS_H
#define ACOS_H

#include "../../export.h"
#include "../../operators.h"

typedef struct AcosDescriptor {
    Device device;
} AcosDescriptor;

typedef AcosDescriptor *infiniopAcosDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAcosDescriptor(infiniopHandle_t handle,
                                                           infiniopAcosDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopAcos(infiniopAcosDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroyAcosDescriptor(infiniopAcosDescriptor_t desc);

#endif
