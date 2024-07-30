#include "../utils.h"
#include "rms_norm.h"

#ifdef ENABLE_CPU
#include "cpu/rms_norm_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/rms_norm.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/rms_norm_cnnl.h"
#include "bang/rms_norm_bang.h"
#endif
#ifdef ENABLE_ASCEND_NPU
#include "ascend/rms_norm_aclnn.h"
#endif

struct RMSNormDescriptor {
    Device device;
};

__C RMSNormDescriptor *createRMSNormDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (RMSNormDescriptor *) (new RMSNormCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return (RMSNormDescriptor *) (new RMSNormCudaDescriptor{device});
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            auto bangDescriptor = new RMSNormBangDescriptor(device);
            bangDescriptor->createCnnlDescriptors();
            return (RMSNormDescriptor *) (bangDescriptor);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            auto ascendDescriptor = new RMSNormAclnnDescriptor(device);
            ascendDescriptor->createAclnnDescriptors();
            return (RMSNormDescriptor *) (ascendDescriptor);
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C void destroyRMSNormDescriptor(RMSNormDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (RMSNormCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (RMSNormCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            auto bangDescriptor = (RMSNormBangDescriptor *) (descriptor);
            bangDescriptor->destroyCnnlDescriptors();
            delete bangDescriptor;
            break;
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            auto ascendDescriptor = (RMSNormAclnnDescriptor *) (descriptor);
            ascendDescriptor->destroyAclnnDescriptors();
            delete ascendDescriptor;
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

__C void rmsNorm(RMSNormDescriptor *descriptor, Tensor y, Tensor x, Tensor w, float epsilon, void *stream) {
    switch (descriptor->device) {
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
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            // Using BANGC Kernel
            // rms_norm_bang_f16(y, x, w, epsilon, stream);
            rms_norm_cnnl_f16((RMSNormBangDescriptor *) (descriptor), y, x, w, epsilon, stream);
            break;
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu:
            rms_norm_aclnn_f16((RMSNormAclnnDescriptor *) (descriptor), y, x, w, epsilon, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}
