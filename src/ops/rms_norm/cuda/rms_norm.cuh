#ifndef __NV_GPU_RMS_NORM_H__
#define __NV_GPU_RMS_NORM_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"

struct RMSNormCudaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t n;
    uint64_t d;
    int64_t stride_y;
    int64_t stride_x;
    DT w_datatype;
    float epsilon;
};

typedef struct RMSNormCudaDescriptor *RMSNormCudaDescriptor_t;

infiniopStatus_t cudaCreateRMSNormDescriptor(CudaHandle_t handle,
                                             RMSNormCudaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y_desc,
                                             infiniopTensorDescriptor_t x_desc,
                                             infiniopTensorDescriptor_t w_desc,
                                             float epsilon);

infiniopStatus_t cudaGetRMSNormWorkspaceSize(RMSNormCudaDescriptor_t desc, uint64_t *size);

infiniopStatus_t cudaRMSNorm(RMSNormCudaDescriptor_t desc,
                             void *workspace,
                             unsigned long int workspace_size,
                             void *y, void const *x, void const *w,
                             void *stream);

infiniopStatus_t cudaDestroyRMSNormDescriptor(RMSNormCudaDescriptor_t desc);

void rms_norm_nv_gpu_f16(RMSNormCudaDescriptor_t desc, void *y, void const *x, void const *w, float epsilon, void *stream);

#endif// __NV_GPU_RMS_NORM_H__
