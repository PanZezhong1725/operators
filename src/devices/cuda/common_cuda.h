#ifndef __COMMON_CUDA_H__
#define __COMMON_CUDA_H__

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_WARP_PER_BLOCK 32
#define WARP_SIZE 32

#define checkCudaErrorWithCode(call, errorCode)          \
    do {                                                 \
        if (auto status = call; status != cudaSuccess) { \
            return errorCode;                            \
        }                                                \
    } while (0)

#define checkCudaError(call) checkCudaErrorWithCode(call, STATUS_BAD_DEVICE)

#define checkCudnnError(call)                                     \
    do {                                                          \
        if (auto status = call; status != CUDNN_STATUS_SUCCESS) { \
            return STATUS_EXECUTION_FAILED;                       \
        }                                                         \
    } while (0)

#include "data_type.h"
#include <cudnn.h>

typedef struct DTCudnnMapping {
    DT layout;
    cudnnDataType_t cudnn_type;
} DTCudnnMapping;

// DT cudnnDataType_t mapping table
constexpr DTCudnnMapping dtMappings[] = {
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

#endif// __COMMON_CUDA_H__
