#ifndef __COMMON_BANG_H__
#define __COMMON_BANG_H__

#include "../../tensor.h"
#include "cnnl.h"
#include <vector>

const int NRAM_MAX_SIZE = 1024 * 256;//the maximum NRAM memory is 1024 * 768
const int GDRAM_MAX_SIZE = 1024 * 1024 * 1024;
inline void setCnnlTensor(cnnlTensorDescriptor_t desc, const TensorLayout* layout) {
    std::vector<int> dims(layout->ndim);
    for (uint64_t i = 0; i < layout->ndim; i++) {
        dims[i] = static_cast<int>(layout->shape[i]);
    }
}

#endif  // __COMMON_BANG_H__
