#include "rms_norm.h"

#ifdef ENABLE_CPU
#include "cpu/rms_norm_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/rms_norm.cuh"
#endif
#ifdef ENABLE_ASCEND_NPU
#include "ascend/rms_norm.h"
#endif

#include "../utils.h"

extern "C" void *createRMSNormDescriptor(Device device, void *config) {
    auto desc = new RMSNormDescriptor{device};
    return (void *) desc;
}

extern "C" void destroyRMSNormDescriptor(void *descriptor) {
    auto desc = (RMSNormDescriptor *) descriptor;
    delete desc;
}

extern "C" void rmsNorm(void *descriptor, MutTensor y, ConstTensor x, ConstTensor w, float epsilon, void *stream) {
    auto desc = (RMSNormDescriptor *) descriptor;
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            rms_norm_cpu_f16(y, x, w, epsilon);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            rms_norm_nv_gpu_f16(y, x, w, epsilon, stream);
            break;
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu:
            rms_norm_ascend_npu_fp16(y, x, w, epsilon, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}
