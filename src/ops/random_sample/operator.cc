#include "../utils.h"
#include "random_sample.h"

#ifdef ENABLE_CPU
#include "cpu/random_sample_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/random_sample.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/random_sample_bang.h"

#endif

struct RandomSampleDescriptor {
    Device device;
};

__C void *createRandomSampleDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return (RandomSampleDescriptor *) (new RandomSampleCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return (RandomSampleDescriptor *) (new RandomSampleCudaDescriptor{device});
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return (RandomSampleDescriptor *) (new RandomSampleBangDescriptor{device});
    }
#endif
    default:
        PANIC(UnsupportedDevice);
    }
    return nullptr;
};

__C void destroyRandomSampleDescriptor(RandomSampleDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (RandomSampleCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (RandomSampleCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (RandomSampleBangDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

__C void random_sample(RandomSampleDescriptor *descriptor, Tensor source, Tensor indices, float topp, int topk, float temperature, void *stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            random_sample_cpu_f16(source, indices, topp, topk, temperature);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            random_sample_nv_gpu_f16(source, indices, topp, topk, temperature, stream);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            
            random_sample_bang_f16(source, indices, topp, topk, temperature, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
};
