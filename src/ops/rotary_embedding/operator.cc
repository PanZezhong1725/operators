#include "../utils.h"
#include "ops/rotary_embedding/rotary_embedding.h"

#ifdef ENABLE_CPU
#include "../../devices/cpu/cpu_handle.h"
#include "cpu/rotary_embedding_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/rotary_embedding.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/rotary_embedding_cnnl.h"
#endif
#ifdef ENABLE_ASCEND_NPU
// #include "ascend/rotary_embedding.h"
#endif

struct RoPEDescriptor {
    Device device;
};


__C infiniopStatus_t infiniopCreateRoPEDescriptor(infiniopHandle_t handle,
                                                  infiniopRoPEDescriptor_t *desc_ptr,
                                                  infiniopTensorDescriptor_t t,
                                                  infiniopTensorDescriptor_t pos_ids,
                                                  infiniopTensorDescriptor_t sin_table,
                                                  infiniopTensorDescriptor_t cos_table) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateRoPEDescriptor((CpuHandle_t) handle, (RoPECpuDescriptor_t *) desc_ptr, t, pos_ids, sin_table, cos_table);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateRoPEDescriptor((CudaHandle_t) handle, (RoPECudaDescriptor_t *) desc_ptr, t, pos_ids, sin_table, cos_table);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetRoPEWorkspaceSize(infiniopRoPEDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetRoPEWorkspaceSize((RoPECpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetRoPEWorkspaceSize((RoPECudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopRoPE(infiniopRoPEDescriptor_t desc,
                                  void *workspace,
                                  uint64_t workspace_size,
                                  void *t,
                                  void const *pos_ids,
                                  void const *sin_table,
                                  void const *cos_table,
                                  void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuRoPE((RoPECpuDescriptor_t) desc, workspace, workspace_size, t, pos_ids, sin_table, cos_table, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaRoPE((RoPECudaDescriptor_t) desc, workspace, workspace_size, t, pos_ids, sin_table, cos_table, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyRoPEDescriptor(infiniopRoPEDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyRoPEDescriptor((RoPECpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyRoPEDescriptor((RoPECudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
