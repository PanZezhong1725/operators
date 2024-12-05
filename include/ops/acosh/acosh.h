#ifndef ACOSH_H
#define ACOSH_H

#include "../../export.h"
#include "../../operators.h"

typedef struct AcoshDescriptor {
    Device device;
} AcoshDescriptor;

typedef AcoshDescriptor *infiniopAcoshDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAcoshDescriptor(infiniopHandle_t handle,
                                                            infiniopAcoshDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y,
                                                            infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopAcosh(infiniopAcoshDescriptor_t desc,
                                            void *y,
                                            void const *x,
                                            void *stream);

__C __export infiniopStatus_t infiniopDestroyAcoshDescriptor(infiniopAcoshDescriptor_t desc);

#endif
