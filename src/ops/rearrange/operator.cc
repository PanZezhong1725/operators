#include "../utils.h"
#include "operators.h"
#include "ops/rearrange/rearrange.h"

#ifdef ENABLE_CPU
#include "cpu/rearrange_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/common_cuda.h"
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/rearrange.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/rearrange_bang.h"
//#include "bang/rearrange_cnnl.h"
#endif
#ifdef ENABLE_ASCEND_NPU
#include "ascend/rearrange_aclnn.h"
#endif

__C infiniopStatus_t infiniopCreateRearrangeDescriptor(
    infiniopHandle_t handle,
    infiniopRearrangeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t src) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateRearrangeDescriptor(handle, (RearrangeCpuDescriptor_t *) desc_ptr, dst, src);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateRearrangeDescriptor((CudaHandle_t) handle, (RearrangeCudaDescriptor_t *) desc_ptr, dst, src);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangCreateRearrangeDescriptor((BangHandle_t) handle, (RearrangeBangDescriptor_t *) desc_ptr, dst, src);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnCreateRearrangeDescriptor((AscendHandle_t) handle,
                                                  (RearrangeAclnnDescriptor_t *) desc_ptr,
                                                  dst,
                                                  src);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopRearrange(infiniopRearrangeDescriptor_t desc, void *dst, void const *src, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuRearrange((RearrangeCpuDescriptor_t) desc, dst, src, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaRearrange((RearrangeCudaDescriptor_t) desc, dst, src, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangRearrange((RearrangeBangDescriptor_t) desc, dst, src, stream);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnRearrange((RearrangeAclnnDescriptor_t) desc,
                                  dst,
                                  src,
                                  stream);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyRearrangeDescriptor(infiniopRearrangeDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyRearrangeDescriptor((RearrangeCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyRearrangeDescriptor((RearrangeCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangDestroyRearrangeDescriptor((RearrangeBangDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnDestroyRearrangeDescriptor((RearrangeAclnnDescriptor_t) desc);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
