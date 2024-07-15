#include "../utils.h"
#include "swiglu.h"

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

struct SwigluDescriptor {
    Device device;
};

__C void *createSwigluDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return (SwigluDescriptor *) (new SwigluCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return (SwigluDescriptor *) (new SwigluCudaDescriptor{device});
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        auto bangDescriptor = new SwigluBangDescriptor(device);
        bangDescriptor->createCnnlDescriptors();
        return (SwigluDescriptor *) (bangDescriptor);
    }
#endif
    default:
        PANIC(UnsupportedDevice);
    }
    return nullptr;
};

__C void destroySwigluDescriptor(SwigluDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (SwigluCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (SwigluCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            auto bangDescriptor = (SwigluBangDescriptor *) (descriptor);
            bangDescriptor->destroyCnnlDescriptors();
            delete bangDescriptor;
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

__C void swiglu(SwigluDescriptor *descriptor, Tensor gate, Tensor up, void *stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            swiglu_cpu_f16(gate, up);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            swiglu_nv_gpu_f16(gate, up, stream);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            // swiglu_cnnl_f16((SwigluBangDescriptor *) (descriptor), gate, up, stream);
            swiglu_bang_f16(gate, up, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
};
