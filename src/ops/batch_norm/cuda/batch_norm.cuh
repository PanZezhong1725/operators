#ifndef __CUDA_BATCH_NORM_H__
#define __CUDA_BATCH_NORM_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct BatchNormCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    std::shared_ptr<Pool<cudnnHandle_t>> cudnn_handles_t;
    const cudnnTensorDescriptor_t x_desc;
    const cudnnTensorDescriptor_t bn_desc;
    const cudnnTensorDescriptor_t y_desc;
    const float alpha;
    const float beta;
    const double eps;
    cudnnBatchNormMode_t mode;
};

typedef struct BatchNormCudaDescriptor *BatchNormCudaDescriptor_t;

infiniopStatus_t cudaCreateBatchNormDescriptor(CudaHandle_t,
                                               BatchNormCudaDescriptor_t *,
                                               infiniopTensorDescriptor_t y,
                                               infiniopTensorDescriptor_t x,
                                               infiniopTensorDescriptor_t scale,
                                               infiniopTensorDescriptor_t b,
                                               infiniopTensorDescriptor_t mean,
                                               infiniopTensorDescriptor_t var,
                                               double eps);

infiniopStatus_t cudaBatchNorm(BatchNormCudaDescriptor_t desc,
                               void *y, void const *x, void const *scale, void const *b,
                               void const *mean, void const *var, void *stream);

infiniopStatus_t cudaDestroyBatchNormDescriptor(BatchNormCudaDescriptor_t desc);

#endif
