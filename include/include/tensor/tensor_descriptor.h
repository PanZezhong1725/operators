#ifndef TENSOR_DESCRIPTOR_H
#define TENSOR_DESCRIPTOR_H

#include "../export.h"
#include "../tensor.h"
#include "../status.h"

__C __export infiniopStatus_t infiniopCreateTensorDescriptor(infiniopTensorDescriptor_t *desc_ptr, uint64_t ndim, uint64_t const *shape_, int64_t const *strides_, DataLayout datatype);

__C __export infiniopStatus_t infiniopDestroyTensorDescriptor(infiniopTensorDescriptor_t desc);

#endif// TENSOR_DESCRIPTOR_H
