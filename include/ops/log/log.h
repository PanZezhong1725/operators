#ifndef LOG_H
#define LOG_H

#include "../../export.h"
#include "../../operators.h"

typedef struct LogDescriptor {
    Device device;
} LogDescriptor;

typedef LogDescriptor *infiniopLogDescriptor_t;

__C __export infiniopStatus_t infiniopCreateLogDescriptor(infiniopHandle_t handle,
                                                          infiniopLogDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y,
                                                          infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopLog(infiniopLogDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyLogDescriptor(infiniopLogDescriptor_t desc);

#endif
