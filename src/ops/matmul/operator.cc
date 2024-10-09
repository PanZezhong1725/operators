#include "../../utils.h"
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
#ifdef ENABLE_ASCEND_NPU
#include "ascend/matmul_aclnn.h"
#endif

__C infiniopStatus_t infiniopCreateMatmulDescriptor(infiniopHandle_t handle,
                                                    infiniopMatmulDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t c_desc,
                                                    float alpha,
                                                    infiniopTensorDescriptor_t a_desc,
                                                    infiniopTensorDescriptor_t b_desc,
                                                    float beta) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateMatmulDescriptor((CpuHandle_t) handle,
                                             (MatmulCpuDescriptor_t *) desc_ptr,
                                             c_desc,
                                             alpha,
                                             a_desc,
                                             b_desc,
                                             beta);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateMatmulDescriptor((CudaHandle_t) handle,
                                              (MatmulCudaDescriptor_t *) desc_ptr,
                                              c_desc,
                                              alpha,
                                              a_desc,
                                              b_desc,
                                              beta);
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnCreateMatmulDescriptor((AscendHandle_t) handle,
                                               (MatmulAclnnDescriptor_t *) desc_ptr,
                                               c_desc,
                                               alpha,
                                               a_desc,
                                               b_desc,
                                               beta,
                                               1);
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
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnGetMatmulWorkspaceSize((MatmulAclnnDescriptor_t) desc,
                                               size);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopMatmul(infiniopMatmulDescriptor_t desc,
                                    void *workspace,
                                    uint64_t workspace_size,
                                    void *c,
                                    void const *a,
                                    void const *b,
                                    void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuMatmul((MatmulCpuDescriptor_t) desc, workspace, workspace_size, c, a, b);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaMatmul((MatmulCudaDescriptor_t) desc, workspace, workspace_size, c, a, b, stream);
#endif
#ifdef ENABLE_CAMBRICON_MLU
            // TODO
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu:
            return aclnnMatmul((MatmulAclnnDescriptor_t) desc,
                               workspace,
                               workspace_size,
                               c,
                               a,
                               b,
                               stream);
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
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnDestroyMatmulDescriptor((MatmulAclnnDescriptor_t) desc);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
