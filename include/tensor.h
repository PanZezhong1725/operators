#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "data_type.h"
#include <stdint.h>

struct TensorLayout {
    struct DataLayout dt;
    uint64_t ndim;
    uint64_t *shape;
    int64_t *strides;
};

typedef struct TensorLayout *TensorDescriptor;

struct TensorTuple {
    TensorDescriptor const layout;
    void *data;
};

typedef struct TensorTuple Tensor;

#endif// __TENSOR_H__
