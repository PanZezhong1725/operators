#ifndef __COMMON_CUDA_H__
#define __COMMON_CUDA_H__

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_WARP_PER_BLOCK 32
#define WARP_SIZE 32

#include <iostream>

#define checkCudaErrorWithCode(call, errorCode)                       \
    do {                                                              \
        if (auto status = call; status != cudaSuccess) {              \
            std::cerr << "CUDA error: " << cudaGetErrorString(status) \
                      << " in file " << __FILE__                      \
                      << ", function " << __func__                    \
                      << ", line " << __LINE__ << std::endl;          \
            return errorCode;                                         \
        }                                                             \
    } while (0)

#define checkCudaError(call) checkCudaErrorWithCode(call, STATUS_BAD_DEVICE)

#define checkCudnnError(call)                                           \
    do {                                                                \
        if (auto status = call; status != CUDNN_STATUS_SUCCESS) {       \
            std::cerr << "CUDNN error: " << cudnnGetErrorString(status) \
                      << " in file " << __FILE__                        \
                      << ", function " << __func__                      \
                      << ", line " << __LINE__ << std::endl;            \
            return STATUS_EXECUTION_FAILED;                             \
        }                                                               \
    } while (0)

#include "data_type.h"
#include <cudnn.h>

typedef struct DTCudnnMapping {
    DT layout;
    cudnnDataType_t cudnn_type;
} DTCudnnMapping;

// DT cudnnDataType_t mapping table
const DTCudnnMapping dtMappings[] = {
    {F16, CUDNN_DATA_HALF},
    {F32, CUDNN_DATA_FLOAT},
    {F64, CUDNN_DATA_DOUBLE},
    {BF16, CUDNN_DATA_BFLOAT16},
    {I8, CUDNN_DATA_INT8},
    {I32, CUDNN_DATA_INT32},
    {I64, CUDNN_DATA_INT64},
    {U8, CUDNN_DATA_UINT8},
};

typedef struct DataLayoutMap {
    int operator[](const DataLayout &layout) const {
        for (const auto &mapping : dtMappings) {
            if (mapping.layout == layout) {
                return mapping.cudnn_type;
            }
        }
        return -1;
    }
} DTMap;

constexpr DTMap dataTypeMap;

// get the corresponding offset in the destination given the flat index of the source (for element mapping in shape broadcast)
inline __device__ uint64_t getDstOffset(uint64_t flat_index, uint64_t ndim, int64_t const *src_strides, int64_t const *dst_strides) {
    uint64_t res = 0;
    for (uint64_t i = 0; i < ndim; ++i) {
        res += flat_index / src_strides[i] * dst_strides[i];
        flat_index %= src_strides[i];
    }
    return res;
}

// get the memory offset of the given element in a tensor given its flat index
inline __device__ uint64_t getOffset(uint64_t flat_index, uint64_t ndim, uint64_t const *shape, int64_t const *strides) {
    uint64_t res = 0;
    for (long i = ndim - 1; i >= 0; --i) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}

#endif// __COMMON_CUDA_H__
