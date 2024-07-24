#include "tensor/tensor_descriptor.h"
#include <cstring>

__C __export void createTensorDescriptor(TensorDescriptor* desc_ptr, uint64_t ndim, uint64_t *shape_, int64_t *strides_, DataLayout datatype) {
    uint64_t *shape = new uint64_t[ndim];
    int64_t *strides = new int64_t[ndim];
    std::memcpy(shape, shape_, ndim * sizeof(uint64_t));
    std::memcpy(strides, strides_, ndim * sizeof(int64_t));
    *desc_ptr = new TensorLayout{datatype, ndim, shape, strides};
}

__C __export void destroyTensorDescriptor(TensorDescriptor desc){
    delete[] desc->shape;
    delete[] desc->strides;
    delete desc;
}
