#ifndef CONV_BIAS_ACT_H
#define CONV_BIAS_ACT_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ConvBiasActDescriptor {
    Device device;
} ConvBiasActDescriptor;

typedef ConvBiasActDescriptor *infiniopConvBiasActDescriptor_t;

__C __export infiniopStatus_t infiniopCreateConvBiasActDescriptor(infiniopHandle_t handle,
                                                                  infiniopConvBiasActDescriptor_t *desc_ptr,
                                                                  infiniopTensorDescriptor_t y,
                                                                  infiniopTensorDescriptor_t x,
                                                                  infiniopTensorDescriptor_t w,
                                                                  infiniopTensorDescriptor_t b,
                                                                  uint64_t const *pads,
                                                                  int64_t const *strides,
                                                                  uint64_t const *dilations,
                                                                  uint64_t n,
                                                                  int activation_mode,
                                                                  double clip_coef = 0.0);

__C __export infiniopStatus_t infiniopGetConvBiasActWorkspaceSize(infiniopConvBiasActDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopConvBiasAct(infiniopConvBiasActDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void const *w, void const *b, void *stream);

__C __export infiniopStatus_t infiniopDestroyConvBiasActDescriptor(infiniopConvBiasActDescriptor_t desc);


#endif
