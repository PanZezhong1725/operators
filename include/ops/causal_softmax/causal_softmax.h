#ifndef CAUSAL_SOFTMAX_H
#define CAUSAL_SOFTMAX_H

#include "../../export.h"
#include "../../operators.h"

typedef struct CausalSoftmaxDescriptor CausalSoftmaxDescriptor;

__C __export CausalSoftmaxDescriptor *createCausalSoftmaxDescriptor(Device, void *config);
__C __export void destroyCausalSoftmaxDescriptor(CausalSoftmaxDescriptor *descriptor);
__C __export void causalSoftmax(CausalSoftmaxDescriptor *descriptor, Tensor y, void *stream);


#endif
