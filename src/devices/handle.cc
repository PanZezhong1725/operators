#include "handle/handle_export.h"
#ifdef ENABLE_CPU
#include "./cpu/cpu_handle.h"
#endif
#ifdef ENABLE_NV_GPU
#include "./cuda/cuda_handle.h"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "./bang/bang_handle.h"
#endif


__C infiniopStatus_t infiniopCreateHandle(infiniopHandle_t *handle_ptr, Device device, int device_id) {
    if (handle_ptr == nullptr) {
        return STATUS_MEMORY_NOT_ALLOCATED;
    }
    if (device_id < 0) {
        return STATUS_BAD_PARAM;
    }

    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return createCpuHandle((CpuHandle_t *) handle_ptr);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return createCudaHandle((CudaHandle_t *) handle_ptr, device_id);
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return createBangHandle((BangHandle_t *) handle_ptr, device_id);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}


__C infiniopStatus_t infiniopDestroyHandle(infiniopHandle_t handle) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete handle;
            return STATUS_SUCCESS;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return deleteCudaHandle((CudaHandle_t) handle);
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (BangHandle_t) handle;
            return STATUS_SUCCESS;
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
