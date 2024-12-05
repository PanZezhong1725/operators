#ifndef COS_H
#define COS_H

#include "../../export.h"
#include "../../operators.h"

typedef struct CosDescriptor {
    Device device;
} CosDescriptor;

typedef CosDescriptor *infiniopCosDescriptor_t;

__C __export infiniopStatus_t infiniopCreateCosDescriptor(infiniopHandle_t handle,
                                                          infiniopCosDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y,
                                                          infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopCos(infiniopCosDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyCosDescriptor(infiniopCosDescriptor_t desc);

#endif
