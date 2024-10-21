#include "../utils.h"
#include "operators.h"
#include "ops/add/add.h"

#ifdef ENABLE_CPU
#include "cpu/add_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/add.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "../../devices/bang/bang_handle.h"
#include "bang/add_bang.h"
#endif

__C infiniopStatus_t infiniopCreateAddDescriptor(
    infiniopHandle_t handle,
    infiniopAddDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c,
    infiniopTensorDescriptor_t a,
    infiniopTensorDescriptor_t b) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateAddDescriptor(handle, (AddCpuDescriptor_t *) desc_ptr, c, a, b);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateAddDescriptor((CudaHandle_t) handle, (AddCudaDescriptor_t *) desc_ptr, c, a, b);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangCreateAddDescriptor((BangHandle_t) handle, (AddBangDescriptor_t *) desc_ptr, c, a, b);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopAdd(infiniopAddDescriptor_t desc, void *c, void const *a, void const *b, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuAdd((AddCpuDescriptor_t) desc, c, a, b, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaAdd((AddCudaDescriptor_t) desc, c, a, b, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangAdd((AddBangDescriptor_t) desc, c, a, b, stream);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyAddDescriptor(infiniopAddDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyAddDescriptor((AddCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyAddDescriptor((AddCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangDestroyAddDescriptor((AddBangDescriptor_t) desc);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
