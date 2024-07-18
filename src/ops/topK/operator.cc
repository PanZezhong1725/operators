#include "../utils.h"
#include "topK.h"

#ifdef ENABLE_CPU
#include "cpu/topK_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/topK_cuda.cuh"
#endif

struct TopKDescriptor {
    Device device;
};

__C void *createTopKDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (TopKDescriptor *) (new TopKCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return (TopKDescriptor *) (new TopKCudaDescriptor{device});
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
};

__C void destroyTopKDescriptor(TopKDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (TopKCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (TopKCudaDescriptor *) (descriptor);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

__C void topK(TopKDescriptor *descriptor, Tensor indices, Tensor probs, Tensor logits, int64_t k, void *stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            topK_cpu(indices, probs, logits, k);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            topK_nv_gpu_f16(indices, probs, logits, k, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
};
