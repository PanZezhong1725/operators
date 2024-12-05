#include "../utils.h"
#include "operators.h"
#include "unary.h"

#ifdef ENABLE_CPU
#include "cpu/unary_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/unary.cuh"
#endif

__C infiniopStatus_t infiniopCreateUnaryDescriptor(
    infiniopHandle_t handle,
    infiniopUnaryDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int mode) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateUnaryDescriptor(handle, (UnaryCpuDescriptor_t *) desc_ptr, y, x, mode);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateUnaryDescriptor((CudaHandle_t) handle, (UnaryCudaDescriptor_t *) desc_ptr, y, x, mode);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopUnary(infiniopUnaryDescriptor_t desc, void *y, void const *x, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuUnary((UnaryCpuDescriptor_t) desc, y, x, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaUnary((UnaryCudaDescriptor_t) desc, y, x, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyUnaryDescriptor(infiniopUnaryDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyUnaryDescriptor((UnaryCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyUnaryDescriptor((UnaryCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}