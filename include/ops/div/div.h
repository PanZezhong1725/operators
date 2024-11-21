#ifndef DIV_H
#define DIV_H

#include "../../export.h"
#include "../../operators.h"

typedef struct DivDescriptor {
    Device device;
} DivDescriptor;

typedef DivDescriptor *infiniopDivDescriptor_t;

__C __export infiniopStatus_t infiniopCreateDivDescriptor(infiniopHandle_t handle,
                                                          infiniopDivDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c,
                                                          infiniopTensorDescriptor_t a,
                                                          infiniopTensorDescriptor_t b);

__C __export infiniopStatus_t infiniopDiv(infiniopDivDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyDivDescriptor(infiniopDivDescriptor_t desc);

#endif
