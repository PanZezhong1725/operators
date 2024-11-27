#include "../utils.h"
#include "operators.h"
#include "ops/conv_bias_act/conv_bias_act.h"

#ifdef ENABLE_CPU
#include "cpu/conv_bias_act_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/conv_bias_act.cuh"
#endif

__C infiniopStatus_t infiniopCreateConvBiasActDescriptor(
    infiniopHandle_t handle,
    infiniopConvBiasActDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t w,
    infiniopTensorDescriptor_t b,
    uint64_t const *pads,
    int64_t const *strides,
    uint64_t const *dilations,
    uint64_t n,
    int activation_mode,
    double clip_coef) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateConvBiasActDescriptor(handle, (ConvBiasActCpuDescriptor_t *) desc_ptr, y, x, w, b, pads, strides, dilations, n, activation_mode, clip_coef);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateConvBiasActDescriptor((CudaHandle_t) handle, (ConvBiasActCudaDescriptor_t *) desc_ptr, y, x, w, b, pads, strides, dilations, n, activation_mode, clip_coef);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetConvBiasActWorkspaceSize(infiniopConvBiasActDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetConvBiasActWorkspaceSize((ConvBiasActCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetConvBiasActWorkspaceSize((ConvBiasActCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopConvBiasAct(infiniopConvBiasActDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void const *w, void const *b, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuConvBiasAct((ConvBiasActCpuDescriptor_t) desc, workspace, workspace_size, y, x, w, b, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaConvBiasAct((ConvBiasActCudaDescriptor_t) desc, workspace, workspace_size, y, x, w, b, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyConvBiasActDescriptor(infiniopConvBiasActDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyConvBiasActDescriptor((ConvBiasActCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyConvBiasActDescriptor((ConvBiasActCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
