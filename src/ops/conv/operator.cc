#include "../utils.h"
#include "operators.h"
#include "ops/conv/conv.h"

#ifdef ENABLE_CPU
#include "cpu/conv_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/conv.cuh"
#endif

__C infiniopStatus_t infiniopCreateConvDescriptor(
    infiniopHandle_t handle,
    infiniopConvDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t w,
    void *pads,
    void *strides,
    void *dilations,
    uint64_t n,
    int device_id) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateConvDescriptor(handle, (ConvCpuDescriptor_t *) desc_ptr, y, x, w, pads, strides, dilations, n);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateConvDescriptor((CudaHandle_t) handle, (ConvCudaDescriptor_t *) desc_ptr, y, x, w, pads, strides, dilations, n, device_id);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetConvWorkspaceSize(infiniopConvDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetConvWorkspaceSize((ConvCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetConvWorkspaceSize((ConvCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopConv(infiniopConvDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void const *w, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuConv((ConvCpuDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaConv((ConvCudaDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyConvDescriptor(infiniopConvDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyConvDescriptor((ConvCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyConvDescriptor((ConvCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
