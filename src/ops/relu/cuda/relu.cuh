#ifndef __CUDA_RELU_H__
#define __CUDA_RELU_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct ReluCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t data_size;
    uint64_t max_grid_size;
};

typedef struct ReluCudaDescriptor *ReluCudaDescriptor_t;

infiniopStatus_t cudaCreateReluDescriptor(CudaHandle_t,
                                          ReluCudaDescriptor_t *,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x);

infiniopStatus_t cudaRelu(ReluCudaDescriptor_t desc,
                          void *y, void const *x,
                          void *stream);

infiniopStatus_t cudaDestroyReluDescriptor(ReluCudaDescriptor_t desc);

#endif
