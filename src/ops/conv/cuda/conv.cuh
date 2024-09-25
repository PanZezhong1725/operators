#ifndef __CUDA_CONV_H__
#define __CUDA_CONV_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cudnn.h>

struct ConvCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    std::shared_ptr<Pool<cudnnHandle_t>> cudnn_handles_t;
    cudnnTensorDescriptor_t const x_desc;
    cudnnFilterDescriptor_t const w_desc;
    cudnnTensorDescriptor_t const y_desc;
    cudnnConvolutionDescriptor_t const op_desc;
    cudnnConvolutionFwdAlgo_t algo;
    const float alpha;
    const float beta;
};

typedef struct ConvCudaDescriptor *ConvCudaDescriptor_t;

infiniopStatus_t cudaCreateConvDescriptor(CudaHandle_t,
                                          ConvCudaDescriptor_t *,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x,
                                          infiniopTensorDescriptor_t w,
                                          void const *pads,
                                          void const *strides,
                                          void const *dilations,
                                          uint64_t n,
                                          int device_id);

infiniopStatus_t cudaGetConvWorkspaceSize(ConvCudaDescriptor_t desc, uint64_t *size);

infiniopStatus_t cudaConv(ConvCudaDescriptor_t desc,
                          void *workspace, uint64_t workspace_size,
                          void *y, void const *x, void const *w,
                          void *stream);

infiniopStatus_t cudaDestroyConvDescriptor(ConvCudaDescriptor_t desc);

#endif
