#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__

#include "../../export.h"
#include "../../operators.h"

typedef struct SoftmaxDescriptor SoftmaxDescriptor;

__C __export SoftmaxDescriptor *createSoftmaxDescriptor(Device, void *config);
__C __export void destroySoftmaxDescriptor(SoftmaxDescriptor *descriptor);
__C __export void softmax(SoftmaxDescriptor *descriptor, Tensor y, void *stream);

#endif
