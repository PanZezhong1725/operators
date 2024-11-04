#include "../utils.h"
#include "operators.h"
#include "ops/pooling/pooling.h"

#ifdef ENABLE_CPU
#include "cpu/pooling_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/common_cuda.h"
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/pooling.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/pooling_bang.h"
//#include "bang/pooling_cnnl.h"
#endif

__C infiniopStatus_t infiniopCreatePoolingDescriptor(
    infiniopHandle_t handle,
    infiniopPoolingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    void const *kernel_shape,
    void const *pads,
    void const *strides,
    uint64_t n,
    int pooling_type) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreatePoolingDescriptor(handle, (PoolingCpuDescriptor_t *) desc_ptr, y, x, kernel_shape, pads, strides, n, pooling_type);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreatePoolingDescriptor((CudaHandle_t) handle, (PoolingCudaDescriptor_t *) desc_ptr, y, x, kernel_shape, pads, strides, n, pooling_type);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangCreatePoolingDescriptor((BangHandle_t) handle, (PoolingBangDescriptor_t *) desc_ptr, y, x, kernel_shape, pads, strides, n, pooling_type);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetPoolingWorkspaceSize(infiniopPoolingDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetPoolingWorkspaceSize((PoolingCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetPoolingWorkspaceSize((PoolingCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangGetPoolingWorkspaceSize((PoolingBangDescriptor_t) desc, size);
        }

#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopPooling(infiniopPoolingDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuPooling((PoolingCpuDescriptor_t) desc, workspace, workspace_size, y, x, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaPooling((PoolingCudaDescriptor_t) desc, workspace, workspace_size, y, x, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangPooling((PoolingBangDescriptor_t) desc, y, x, stream);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyPoolingDescriptor(infiniopPoolingDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyPoolingDescriptor((PoolingCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyPoolingDescriptor((PoolingCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangDestroyPoolingDescriptor((PoolingBangDescriptor_t) desc);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
