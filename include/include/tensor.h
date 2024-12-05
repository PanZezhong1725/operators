#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "data_type.h"
#include <stdint.h>

struct TensorDescriptor {
    // Datatype
    DT dt;
    // Number of dimensions
    uint64_t ndim;
    // Shape of the tensor, ndim elements
    uint64_t *shape;
    // Stride of each dimension in elements, ndim elements
    int64_t *strides;
};

typedef struct TensorDescriptor *infiniopTensorDescriptor_t;

#endif// __TENSOR_H__
