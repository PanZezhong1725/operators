#ifndef SUB_H
#define SUB_H

#include "../../export.h"
#include "../../operators.h"

typedef struct SubDescriptor {
    Device device;
} SubDescriptor;

typedef SubDescriptor *infiniopSubDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSubDescriptor(infiniopHandle_t handle,
                                                          infiniopSubDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c,
                                                          infiniopTensorDescriptor_t a,
                                                          infiniopTensorDescriptor_t b);

__C __export infiniopStatus_t infiniopSub(infiniopSubDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroySubDescriptor(infiniopSubDescriptor_t desc);

#endif
