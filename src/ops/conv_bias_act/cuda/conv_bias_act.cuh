#ifndef __CUDA_CONV_BIAS_ACT_H__
#define __CUDA_CONV_BIAS_ACT_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "../conv_bias_act_common.h"
#include <cudnn.h>

struct ConvBiasActCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    std::shared_ptr<Pool<cudnnHandle_t>> cudnn_handles_t;
    cudnnTensorDescriptor_t const x_desc;
    cudnnFilterDescriptor_t const w_desc;
    cudnnTensorDescriptor_t const b_desc;
    cudnnTensorDescriptor_t const y_desc;
    cudnnConvolutionDescriptor_t const op_desc;
    cudnnActivationDescriptor_t const act_desc;
    cudnnConvolutionFwdAlgo_t algo;
    const float alpha;
    const float beta;
    uint64_t workspace_size;
};

typedef struct ConvBiasActCudaDescriptor *ConvBiasActCudaDescriptor_t;

infiniopStatus_t cudaCreateConvBiasActDescriptor(CudaHandle_t,
                                                 ConvBiasActCudaDescriptor_t *,
                                                 infiniopTensorDescriptor_t y,
                                                 infiniopTensorDescriptor_t x,
                                                 infiniopTensorDescriptor_t w,
                                                 infiniopTensorDescriptor_t b,
                                                 uint64_t const *pads,
                                                 int64_t const *strides,
                                                 uint64_t const *dilations,
                                                 uint64_t n,
                                                 int activation_mode,
                                                 double clip_coef);

infiniopStatus_t cudaGetConvBiasActWorkspaceSize(ConvBiasActCudaDescriptor_t desc, uint64_t *size);

infiniopStatus_t cudaConvBiasAct(ConvBiasActCudaDescriptor_t desc,
                                 void *workspace, uint64_t workspace_size,
                                 void *y, void const *x, void const *w,
                                 void const *b, void *stream);

infiniopStatus_t cudaDestroyConvBiasActDescriptor(ConvBiasActCudaDescriptor_t desc);

#endif
