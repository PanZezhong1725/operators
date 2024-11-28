#include "../utils.h"
#include "binary.h"
#include "operators.h"

#ifdef ENABLE_CPU
#include "cpu/binary_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/binary.cuh"
#endif

__C infiniopStatus_t infiniopCreateBinaryDescriptor(
    infiniopHandle_t handle,
    infiniopBinaryDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c,
    infiniopTensorDescriptor_t a,
    infiniopTensorDescriptor_t b,
    int mode) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateBinaryDescriptor(handle, (BinaryCpuDescriptor_t *) desc_ptr, c, a, b, mode);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateBinaryDescriptor((CudaHandle_t) handle, (BinaryCudaDescriptor_t *) desc_ptr, c, a, b, mode);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopBinary(infiniopBinaryDescriptor_t desc, void *c, void const *a, void const *b, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuBinary((BinaryCpuDescriptor_t) desc, c, a, b, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaBinary((BinaryCudaDescriptor_t) desc, c, a, b, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyBinaryDescriptor(infiniopBinaryDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyBinaryDescriptor((BinaryCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyBinaryDescriptor((BinaryCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
