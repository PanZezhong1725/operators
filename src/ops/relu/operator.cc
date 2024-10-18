#include "../utils.h"
#include "operators.h"
#include "ops/relu/relu.h"

#ifdef ENABLE_CPU
#include "cpu/relu_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/relu.cuh"
#endif

__C infiniopStatus_t infiniopCreateReluDescriptor(
    infiniopHandle_t handle,
    infiniopReluDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateReluDescriptor(handle, (ReluCpuDescriptor_t *) desc_ptr, y, x);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateReluDescriptor((CudaHandle_t) handle, (ReluCudaDescriptor_t *) desc_ptr, y, x);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopRelu(infiniopReluDescriptor_t desc, void *y, void const *x, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuRelu((ReluCpuDescriptor_t) desc, y, x, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaRelu((ReluCudaDescriptor_t) desc, y, x, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyReluDescriptor(infiniopReluDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyReluDescriptor((ReluCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyReluDescriptor((ReluCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
