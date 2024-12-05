#ifndef ATANH_H
#define ATANH_H

#include "../../export.h"
#include "../../operators.h"

typedef struct AtanhDescriptor {
    Device device;
} AtanhDescriptor;

typedef AtanhDescriptor *infiniopAtanhDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAtanhDescriptor(infiniopHandle_t handle,
                                                            infiniopAtanhDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y,
                                                            infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopAtanh(infiniopAtanhDescriptor_t desc,
                                            void *y,
                                            void const *x,
                                            void *stream);

__C __export infiniopStatus_t infiniopDestroyAtanhDescriptor(infiniopAtanhDescriptor_t desc);

#endif
