#include "../utils.h"
#include "ops/causal_softmax/causal_softmax.h"

#ifdef ENABLE_CPU
#include "cpu/causal_softmax_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/common_cuda.h"
#include "cuda/causal_softmax.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/causal_softmax_cnnl.h"
#include "bang/causal_softmax_bang.h"
#endif

struct CausalSoftmaxDescriptor {
    Device device;
};

__C CausalSoftmaxDescriptor *createCausalSoftmaxDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (CausalSoftmaxDescriptor *) (new CausalSoftmaxCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (CausalSoftmaxDescriptor *) (new CausalSoftmaxCudaDescriptor{device});
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return (CausalSoftmaxDescriptor *) (new CausalSoftmaxBangDescriptor(device));
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C void destroyCausalSoftmaxDescriptor(CausalSoftmaxDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (CausalSoftmaxCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (CausalSoftmaxCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (CausalSoftmaxBangDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

__C void causalSoftmax(CausalSoftmaxDescriptor *descriptor, Tensor y, void *stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            causal_softmax_cpu_f16(y);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            causal_softmax_nv_gpu_f16((CausalSoftmaxCudaDescriptor *) descriptor, y, stream);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            // causal_softmax_bang_f16(y, y, stream);
            causal_softmax_cnnl_f16(y, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}
