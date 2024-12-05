#ifndef SINH_H
#define SINH_H

#include "../../export.h"
#include "../../operators.h"

typedef struct SinhDescriptor {
    Device device;
} SinhDescriptor;

typedef SinhDescriptor *infiniopSinhDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSinhDescriptor(infiniopHandle_t handle,
                                                           infiniopSinhDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopSinh(infiniopSinhDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroySinhDescriptor(infiniopSinhDescriptor_t desc);

#endif
