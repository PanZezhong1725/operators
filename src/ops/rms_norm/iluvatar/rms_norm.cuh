﻿#ifndef __NV_GPU_RMS_NORM_H__
#define __NV_GPU_RMS_NORM_H__

#include "../../../../include/operators.h"
#include "../../../devices/iluvatar/ilu_handle.h"

struct RMSNormCudaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    unsigned long int n;
    unsigned long int d;
    unsigned long int stride_y;
    unsigned long int stride_x;
    DT w_datatype;
    float epsilon;
};

typedef struct RMSNormCudaDescriptor *RMSNormCudaDescriptor_t;

infiniopStatus_t cudaCreateRMSNormDescriptor(IluHandle_t handle,
                                             RMSNormCudaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y_desc,
                                             infiniopTensorDescriptor_t x_desc,
                                             infiniopTensorDescriptor_t w_desc,
                                             float epsilon);

infiniopStatus_t cudaGetRMSNormWorkspaceSize(RMSNormCudaDescriptor_t desc, unsigned long int *size);

infiniopStatus_t cudaRMSNorm(RMSNormCudaDescriptor_t desc,
                             void *workspace,
                             unsigned long int workspace_size,
                             void *y, void const *x, void const *w,
                             void *stream);

infiniopStatus_t cudaDestroyRMSNormDescriptor(RMSNormCudaDescriptor_t desc);

void rms_norm_nv_gpu_f16(RMSNormCudaDescriptor_t desc, void *y, void const *x, void const *w, float epsilon, void *stream);

#endif// __NV_GPU_RMS_NORM_H__
