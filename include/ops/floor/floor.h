#ifndef FLOOR_H
#define FLOOR_H

#include "../../export.h"
#include "../../operators.h"

typedef struct FloorDescriptor {
    Device device;
} FloorDescriptor;

typedef FloorDescriptor *infiniopFloorDescriptor_t;

__C __export infiniopStatus_t infiniopCreateFloorDescriptor(infiniopHandle_t handle,
                                                            infiniopFloorDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y,
                                                            infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopFloor(infiniopFloorDescriptor_t desc,
                                            void *y,
                                            void const *x,
                                            void *stream);

__C __export infiniopStatus_t infiniopDestroyFloorDescriptor(infiniopFloorDescriptor_t desc);

#endif
