#ifndef SWIGLU_H
#define SWIGLU_H

#include "../../export.h"
#include "../../operators.h"

typedef struct SwigluDescriptor SwigluDescriptor;

__C __export void *createSwigluDescriptor(Device, void *config);
__C __export void destroySwigluDescriptor(SwigluDescriptor *descriptor);
__C __export void swiglu(SwigluDescriptor *descriptor, Tensor gate, Tensor up, void *stream);

#endif
