#include "../../utils.h"
#include "operators.h"
#include "ops/rms_norm/rms_norm.h"

#ifdef ENABLE_CPU
#include "cpu/rms_norm_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/common_cuda.h"
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/rms_norm.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "../../devices/bang/bang_handle.h"
#include "bang/rms_norm_bang.h"
#include "bang/rms_norm_cnnl.h"
#endif
#ifdef ENABLE_ASCEND_NPU
#include "ascend/rms_norm_aclnn.h"
#endif

__C infiniopStatus_t infiniopCreateRMSNormDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateRMSNormDescriptor(handle, (RMSNormCpuDescriptor_t *) desc_ptr, y_desc, x_desc, w_desc, epsilon);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateRMSNormDescriptor((CudaHandle_t) handle, (RMSNormCudaDescriptor_t *) desc_ptr, y_desc, x_desc, w_desc, epsilon);
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangCreateRMSNormDescriptor((BangHandle_t) handle, (RMSNormBangDescriptor_t *) desc_ptr, y_desc);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnCreateRMSNormDescriptor((AscendHandle_t) handle,
                                                (RMSNormAclnnDescriptor_t *) desc_ptr,
                                                y_desc,
                                                x_desc,
                                                w_desc,
                                                epsilon);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetRMSNormWorkspaceSize(infiniopRMSNormDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetRMSNormWorkspaceSize((RMSNormCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetRMSNormWorkspaceSize((RMSNormCudaDescriptor_t) desc, size);
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangGetRMSNormWorkspaceSize((RMSNormBangDescriptor_t) desc, size);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnGetRMSNormWorkspaceSize((RMSNormAclnnDescriptor_t) desc,
                                                size);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopRMSNorm(infiniopRMSNormDescriptor_t desc, void *workspace, uint64_t workspace_size,
                                     void *y, void *x, void *w, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuRMSNorm((RMSNormCpuDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaRMSNorm((RMSNormCudaDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangRMSNorm((RMSNormBangDescriptor_t) desc, workspace, workspace_size, data, stream);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnRMSNorm((RMSNormAclnnDescriptor_t) desc,
                                workspace,
                                workspace_size,
                                y,
                                x,
                                w,
                                stream);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyRMSNormDescriptor(infiniopRMSNormDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyRMSNormDescriptor((RMSNormCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyRMSNormDescriptor((RMSNormCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangDestroyRMSNormDescriptor((RMSNormBangDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnDestroyRMSNormDescriptor((RMSNormAclnnDescriptor_t) desc);
        }

#endif
    }
    return STATUS_BAD_DEVICE;
}
