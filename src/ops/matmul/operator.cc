#include "../utils.h"
#include "ops/matmul/matmul.h"

#ifdef ENABLE_CPU
#include "cpu/matmul_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/matmul_cuda.h"
#include <cublas_v2.h>
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/matmul_cnnl.h"
#endif

__C infiniopStatus_t infiniopCreateMatmulDescriptor(infiniopHandle_t handle,
                                                    infiniopMatmulDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t c_desc,
                                                    infiniopTensorDescriptor_t a_desc,
                                                    infiniopTensorDescriptor_t b_desc) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateMatmulDescriptor((CpuHandle_t) handle, (MatmulCpuDescriptor_t *) desc_ptr, c_desc, a_desc, b_desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateMatmulDescriptor((CudaHandle_t) handle, (MatmulCudaDescriptor_t *) desc_ptr, c_desc, a_desc, b_desc);
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetMatmulWorkspaceSize(infiniopMatmulDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetMatmulWorkspaceSize((MatmulCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetMatmulWorkspaceSize((MatmulCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopMatmul(infiniopMatmulDescriptor_t desc, void *workspace, uint64_t workspace_size, void *c, void const *a, void const *b, float alpha, float beta, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuMatmul((MatmulCpuDescriptor_t) desc, workspace, workspace_size, c, beta, a, b, alpha);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaMatmul((MatmulCudaDescriptor_t) desc, workspace, workspace_size, c, beta, a, b, alpha, stream);
#endif
#ifdef ENABLE_CAMBRICON_MLU
            // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

infiniopStatus_t infiniopDestroyMatmulDescriptor(infiniopMatmulDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyMatmulDescriptor((MatmulCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyMatmulDescriptor((MatmulCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
