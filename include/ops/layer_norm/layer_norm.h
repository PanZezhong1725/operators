#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "../../export.h"
#include "../../operators.h"

typedef struct LayerNormDescriptor {
    Device device;
} LayerNormDescriptor;

typedef LayerNormDescriptor *infiniopLayerNormDescriptor_t;

__C __export infiniopStatus_t infiniopCreateLayerNormDescriptor(
    infiniopHandle_t handle,
    infiniopLayerNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t y_desc,
    float epsilon);



__C __export infiniopStatus_t infiniopLayerNorm(infiniopLayerNormDescriptor_t desc, 
                                              void const *x, void const *w, void const *b, void *y, void *stream);

__C __export infiniopStatus_t infiniopDestroyLayerNormDescriptor(infiniopLayerNormDescriptor_t desc);

#endif
