#include "../utils.h"
#include "ops/rotary_embedding/rotary_embedding.h"

#ifdef ENABLE_CPU
#include "cpu/rotary_embedding_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/rotary_embedding.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/rotary_embedding_cnnl.h"
#endif

struct RotaryEmbeddingDescriptor {
    Device device;
};

__C void *createRotaryEmbeddingDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (RotaryEmbeddingDescriptor *) (new RotaryEmbeddingCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return (RotaryEmbeddingDescriptor *) (new RotaryEmbeddingCudaDescriptor{device});
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            auto bangDescriptor = new RotaryEmbeddingBangDescriptor(device);
            bangDescriptor->createCnnlDescriptors();
            return (RotaryEmbeddingDescriptor *) (bangDescriptor);
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
};

__C void destroyRotaryEmbeddingDescriptor(RotaryEmbeddingDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (RotaryEmbeddingCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (RotaryEmbeddingCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            auto bangDescriptor = (RotaryEmbeddingBangDescriptor *) (descriptor);
            bangDescriptor->destroyCnnlDescriptors();
            delete bangDescriptor;
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

__C void rotaryEmbedding(RotaryEmbeddingDescriptor *descriptor, Tensor t, Tensor pos, float theta, void *stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            rotary_embedding_cpu_f16(t, pos, theta);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            rotary_embedding_nv_gpu_f16(t, pos, theta, stream);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            rotary_embedding_cnnl_f16((RotaryEmbeddingBangDescriptor *) (descriptor), t, pos, theta, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
};
