#ifndef ROUND_H
#define ROUND_H

#include "../../export.h"
#include "../../operators.h"

typedef struct RoundDescriptor {
    Device device;
} RoundDescriptor;

typedef RoundDescriptor *infiniopRoundDescriptor_t;

__C __export infiniopStatus_t infiniopCreateRoundDescriptor(infiniopHandle_t handle,
                                                            infiniopRoundDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y,
                                                            infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopRound(infiniopRoundDescriptor_t desc,
                                            void *y,
                                            void const *x,
                                            void *stream);

__C __export infiniopStatus_t infiniopDestroyRoundDescriptor(infiniopRoundDescriptor_t desc);

#endif
