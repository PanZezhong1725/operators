#ifndef TENSOR_DESCRIPTOR_H
#define TENSOR_DESCRIPTOR_H

#include "../export.h"
#include "../tensor.h"

__C __export void createTensorDescriptor(TensorDescriptor* desc_ptr, uint64_t ndim, uint64_t *shape_, int64_t *strides_, DataLayout datatype);

__C __export void destroyTensorDescriptor(TensorDescriptor desc);

#endif// TENSOR_DESCRIPTOR_H
