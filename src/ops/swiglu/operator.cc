#include "../../utils.h"
#include "operators.h"
#include "ops/swiglu/swiglu.h"

#ifdef ENABLE_CPU
#include "cpu/swiglu_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/swiglu.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/swiglu_bang.h"
#include "bang/swiglu_cnnl.h"
#endif
#ifdef ENABLE_ASCEND_NPU
// #include "ascend/swiglu.h"
#include "ascend/swiglu_aclnn.h"
#endif

__C infiniopStatus_t infiniopCreateSwiGLUDescriptor(infiniopHandle_t handle,
                                                    infiniopSwiGLUDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t c_desc,
                                                    infiniopTensorDescriptor_t a_desc,
                                                    infiniopTensorDescriptor_t b_desc) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateSwiGLUDescriptor(handle, (SwiGLUCpuDescriptor_t *) desc_ptr, c_desc, a_desc, b_desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaCreateSwiGLUDescriptor((CudaHandle_t) handle, (SwiGLUCudaDescriptor_t *) desc_ptr, c_desc, a_desc, b_desc);
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangCreateSwiGLUDescriptor((BangHandle_t) handle,
                                              (SwiGLUBangDescriptor_t *) desc_ptr,
                                              c_desc,
                                              a_desc,
                                              b_desc);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu:
            // return ascendCreateSwiGLUDescriptor(handle, (SwiGLUAscendDescriptor_t *) desc_ptr, c_desc, a_desc, b_desc);
            return aclnnCreateSwiGLUDescriptor((AscendHandle_t) handle, (SwiGLUAclnnDescriptor_t *) desc_ptr, c_desc, a_desc, b_desc);
#endif
    }
    return STATUS_BAD_DEVICE;
};

__C infiniopStatus_t infiniopSwiGLU(infiniopSwiGLUDescriptor_t desc,
                                    void *c,
                                    void const *a,
                                    void const *b,
                                    void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuSwiGLU((SwiGLUCpuDescriptor_t) desc, c, a, b, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaSwiGLU((SwiGLUCudaDescriptor_t) desc, c, a, b, stream);
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangSwiGLU((SwiGLUBangDescriptor_t) desc, c, a, b, stream);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu:
            // return ascendSwiGLU((SwiGLUAscendDescriptor_t) desc, c, a, b, stream);
            return aclnnSwiGLU((SwiGLUAclnnDescriptor_t) desc, c, a, b, stream);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroySwiGLUDescriptor(infiniopSwiGLUDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroySwiGLUDescriptor((SwiGLUCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaDestroySwiGLUDescriptor((SwiGLUCudaDescriptor_t) desc);
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangDestroySwiGLUDescriptor((SwiGLUBangDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu:
            // return ascendDestroySwiGLUDescriptor((SwiGLUAscendDescriptor_t) desc);
            return aclnnDestroySwiGLUDescriptor((SwiGLUAclnnDescriptor_t) desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}
