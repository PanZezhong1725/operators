#include "../utils.h"
#include "operators.h"
#include "ops/batch_norm/batch_norm.h"

#ifdef ENABLE_CPU
#include "cpu/batch_norm_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/batch_norm.cuh"
#endif

__C infiniopStatus_t infiniopCreateBatchNormDescriptor(
    infiniopHandle_t handle,
    infiniopBatchNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t scale,
    infiniopTensorDescriptor_t b,
    infiniopTensorDescriptor_t mean,
    infiniopTensorDescriptor_t var,
    double eps) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateBatchNormDescriptor(handle, (BatchNormCpuDescriptor_t *) desc_ptr, y, x, scale, b, mean, var, eps);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateBatchNormDescriptor((CudaHandle_t) handle, (BatchNormCudaDescriptor_t *) desc_ptr, y, x, scale, b, mean, var, eps);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopBatchNorm(infiniopBatchNormDescriptor_t desc, void *y, void const *x, void const *scale, void const *b,
                                       void const *mean, void const *var, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuBatchNorm((BatchNormCpuDescriptor_t) desc, y, x, scale, b, mean, var, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaBatchNorm((BatchNormCudaDescriptor_t) desc, y, x, scale, b, mean, var, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyBatchNormDescriptor(infiniopBatchNormDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyBatchNormDescriptor((BatchNormCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyBatchNormDescriptor((BatchNormCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
