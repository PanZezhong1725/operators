#include "../utils.h"
#include "operators.h"
#include "ops/layer_norm/layer_norm.h"

#ifdef ENABLE_CPU
#include "cpu/layer_norm_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/common_cuda.h"
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/layer_norm.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "../../devices/bang/bang_handle.h"
#include "bang/layer_norm_bang.h"
#endif

__C infiniopStatus_t infiniopCreateLayerNormDescriptor(
    infiniopHandle_t handle,
    infiniopLayerNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t y_desc,
    float epsilon) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateLayerNormDescriptor(handle, (LayerNormCpuDescriptor_t *) desc_ptr, x_desc, w_desc, b_desc, y_desc, epsilon);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateLayerNormDescriptor((CudaHandle_t) handle, (LayerNormCudaDescriptor_t *) desc_ptr,x_desc, w_desc, b_desc, y_desc, epsilon);
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangCreateLayerNormDescriptor((BangHandle_t) handle, (LayerNormBangDescriptor_t *) desc_ptr, x_desc, w_desc, b_desc, y_desc, epsilon);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopLayerNorm(infiniopLayerNormDescriptor_t desc,
                                     void const *x, void const *w, void const *b, void *y, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuLayerNorm((LayerNormCpuDescriptor_t) desc, x, w, b, y, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaLayerNorm((LayerNormCudaDescriptor_t) desc, x, w, b, y, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangLayerNorm((LayerNormBangDescriptor_t) desc, x, w, b, y, stream);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyLayerNormDescriptor(infiniopLayerNormDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyLayerNormDescriptor((LayerNormCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyLayerNormDescriptor((LayerNormCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangDestroyLayerNormDescriptor((LayerNormBangDescriptor_t) desc);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
