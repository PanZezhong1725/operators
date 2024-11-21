#ifndef MOD_H
#define MOD_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ModDescriptor {
    Device device;
} ModDescriptor;

typedef ModDescriptor *infiniopModDescriptor_t;

__C __export infiniopStatus_t infiniopCreateModDescriptor(infiniopHandle_t handle,
                                                          infiniopModDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c,
                                                          infiniopTensorDescriptor_t a,
                                                          infiniopTensorDescriptor_t b);

__C __export infiniopStatus_t infiniopMod(infiniopModDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyModDescriptor(infiniopModDescriptor_t desc);

#endif
