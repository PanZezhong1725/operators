#ifndef CONV_ACT_H
#define CONV_ACT_H

#include "../../export.h"
#include "../../operators.h"
#include <cstddef>

/**
 * @brief Specifies the type of activation function
 */
struct ActivationMode {

    enum Mode {
        // activation functions
        IDENTITY,
        RELU,
        SIGMOID,

        // Count
        // NOTE: new activation functions should add before "Count"
        Count,
    };
    constexpr static size_t numOfActivationFunctions = Mode::Count;
};

typedef struct ConvActDescriptor {
    Device device;
} ConvActDescriptor;

typedef ConvActDescriptor *infiniopConvActDescriptor_t;

__C __export infiniopStatus_t infiniopCreateConvActDescriptor(infiniopHandle_t handle,
                                                              infiniopConvActDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t y,
                                                              infiniopTensorDescriptor_t x,
                                                              infiniopTensorDescriptor_t w,
                                                              infiniopTensorDescriptor_t b,
                                                              uint64_t const *pads,
                                                              int64_t const *strides,
                                                              uint64_t const *dilations,
                                                              uint64_t n,
                                                              ActivationMode::Mode activation_mode,
                                                              double clip_coef = 0.0);

__C __export infiniopStatus_t infiniopGetConvActWorkspaceSize(infiniopConvActDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopConvAct(infiniopConvActDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void const *w, void const *b, void *stream);

__C __export infiniopStatus_t infiniopDestroyConvActDescriptor(infiniopConvActDescriptor_t desc);


#endif
