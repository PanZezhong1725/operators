#include "../utils.h"
#include "ops/reform/reform.h"

#ifdef ENABLE_CPU
#include "cpu/reform_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/reform.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/reform_bang.h"
#endif

struct ReformDescriptor {
    Device device;
};

__C ReformDescriptor *createReformDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (ReformDescriptor *) (new ReformCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (ReformDescriptor *) (new ReformCudaDescriptor{device});
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return (ReformDescriptor *) (new ReformBangDescriptor{device});
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C void destroyReformDescriptor(ReformDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (ReformCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (ReformCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (ReformBangDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

__C void reform(ReformDescriptor *descriptor, Tensor y, Tensor x, void *stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            reform_cpu(y, x);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            reform_nv_gpu(y, x, stream);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            reform_bang(y, x, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
};
