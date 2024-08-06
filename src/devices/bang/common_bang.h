#ifndef __COMMON_BANG_H__
#define __COMMON_BANG_H__

#include "cnnl.h"
#include "tensor.h"
#include <iostream>
#include <vector>

const int NRAM_MAX_SIZE = 1024 * 256;//the maximum NRAM memory is 1024 * 768
const int GDRAM_MAX_SIZE = 1024 * 1024 * 1024;

// set cnnl tensor descriptor without strides11
inline void setCnnlTensor(cnnlTensorDescriptor_t desc, const TensorLayout* layout) {
    std::vector<int> dims(layout->ndim);
    for (uint64_t i = 0; i < layout->ndim; i++) {
        dims[i] = static_cast<int>(layout->shape[i]);
    }
    cnnlSetTensorDescriptor(desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
                            dims.size(), dims.data());
}

// set cnnl tensor descriptor with strides
inline void setCnnlTensorEx(cnnlTensorDescriptor_t desc, const TensorLayout *layout) {
    std::vector<int> dim_size(layout->ndim), dim_stride(layout->ndim);
    for (uint64_t i = 0; i < layout->ndim; i++) {
        dim_size[i] = static_cast<int>(layout->shape[i]);
        dim_stride[i] = static_cast<int>(layout->strides[i] / layout->dt.size);
    }
    cnnlSetTensorDescriptorEx(desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
                              dim_size.size(), dim_size.data(), dim_stride.data());
}

#endif  // __COMMON_BANG_H__
