#include "../utils.h"
#include "operators.h"
#include "ops/causal_softmax/causal_softmax.h"

#ifdef ENABLE_CPU
#include "cpu/causal_softmax_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/common_cuda.h"
#include "cuda/causal_softmax.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/causal_softmax_bang.h"
#include "bang/causal_softmax_cnnl.h"
#endif

__C infiniopStatus_t infiniopCreateCausalSoftmaxDescriptor(
    infiniopHandle_t handle,
    infiniopCausalSoftmaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateCausalSoftmaxDescriptor(handle, (CausalSoftmaxCpuDescriptor_t *) desc_ptr, y_desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateCausalSoftmaxDescriptor(handle, (CausalSoftmaxCudaDescriptor_t *) desc_ptr, y_desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetCausalSoftmaxWorkspaceSize(infiniopCausalSoftmaxDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetCausalSoftmaxWorkspaceSize((CausalSoftmaxCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetCausalSoftmaxWorkspaceSize((CausalSoftmaxCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopCausalSoftmax(infiniopCausalSoftmaxDescriptor_t desc, void *workspace, uint64_t workspace_size, void *data, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCausalSoftmax((CausalSoftmaxCpuDescriptor_t) desc, workspace, workspace_size, data, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCausalSoftmax((CausalSoftmaxCudaDescriptor_t) desc, workspace, workspace_size, data, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyCausalSoftmaxDescriptor(infiniopCausalSoftmaxDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyCausalSoftmaxDescriptor((CausalSoftmaxCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyCausalSoftmaxDescriptor((CausalSoftmaxCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
