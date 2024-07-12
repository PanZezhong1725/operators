#include "../utils.h"
#include "softmax.h"

#ifdef ENABLE_CPU
#include "cpu/softmax_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/softmax_cuda.cuh"
#include "../../devices/cuda/common_cuda.h"
#endif

struct SoftmaxDescriptor {
    Device device;
};

__C SoftmaxDescriptor *createSoftmaxDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (SoftmaxDescriptor *) (new SoftmaxCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            ASSERT_VALID_PTR(config);
            SoftmaxCudaConfig *cuda_config = (SoftmaxCudaConfig *) config;
            return (SoftmaxDescriptor *) (new SoftmaxCudaDescriptor{
                device,
                ROUND_UP_DIV(cuda_config->max_dim, MAX_THREADS_PER_BLOCK)});
        }

#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C void destroySoftmaxDescriptor(SoftmaxDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (SoftmaxCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (SoftmaxCudaDescriptor *) (descriptor);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

__C void softmax(SoftmaxDescriptor *descriptor, Tensor y, void *stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            softmax_cpu_f16(y);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            softmax_nv_gpu_f16((softmaxCudaDescriptor *) descriptor, y, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}
