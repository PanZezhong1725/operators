#include "../utils.h"
#include "operators.h"
#include "ops/transpose/transpose.h"

#ifdef ENABLE_CPU
#include "cpu/transpose_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/transpose.cuh"
#endif

__C infiniopStatus_t infiniopCreateTransposeDescriptor(
    infiniopHandle_t handle,
    infiniopTransposeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    uint64_t const *perm,
    uint64_t n) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateTransposeDescriptor(handle, (TransposeCpuDescriptor_t *) desc_ptr, y, x, perm, n);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateTransposeDescriptor((CudaHandle_t) handle, (TransposeCudaDescriptor_t *) desc_ptr, y, x, perm, n);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopTranspose(infiniopTransposeDescriptor_t desc, void *y, void const *x, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuTranspose((TransposeCpuDescriptor_t) desc, y, x, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaTranspose((TransposeCudaDescriptor_t) desc, y, x, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyTransposeDescriptor(infiniopTransposeDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyTransposeDescriptor((TransposeCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyTransposeDescriptor((TransposeCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
