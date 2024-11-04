#ifndef __CUDA_POOLING_H__
#define __CUDA_POOLING_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"

struct PoolingCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    std::shared_ptr<Pool<cudnnHandle_t>> cudnn_handles_t;
    cudnnTensorDescriptor_t const x_desc;
    cudnnTensorDescriptor_t const y_desc;
    cudnnPoolingDescriptor_t const pool_desc;
    const float alpha;
    const float beta;
};

typedef struct PoolingCudaDescriptor *PoolingCudaDescriptor_t;

infiniopStatus_t cudaCreatePoolingDescriptor(CudaHandle_t handle,
                                             PoolingCudaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y,
                                             infiniopTensorDescriptor_t x,
                                             void const *kernel_shape,
                                             void const *pads,
                                             void const *strides,
                                             uint64_t n,
                                             int pooling_type);

infiniopStatus_t cudaPooling(PoolingCudaDescriptor_t desc,
                             void *y,
                             void const *x,
                             void *stream);

infiniopStatus_t cudaDestroyPoolingDescriptor(PoolingCudaDescriptor_t desc);

inline cudnnPoolingMode_t getPoolingMode(int pooling_type) {
    switch (pooling_type) {
        case 0:
            return CUDNN_POOLING_MAX;
        case 1:
            return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        default:
            return CUDNN_POOLING_MAX;
    }
}

#endif// __CUDA_POOLING_H__
