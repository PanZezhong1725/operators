#include "../utils.h"
#include "matmul.h"

#ifdef ENABLE_CPU
#include "cpu/matmul_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/matmul_cuda.h"
#include <cublas_v2.h>
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/matmul_cnnl.h"
#endif
#ifdef ENABLE_ASCEND_NPU
#include "ascend/matmul_aclnn.h"
#endif

struct MatmulDescriptor {
    Device device;
};

__C MatmulDescriptor *createMatmulDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (MatmulDescriptor *) (new MatmulCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (MatmulDescriptor *) (new MatmulCudaDescriptor(device));
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            auto bangDescriptor = new MatmulBangDescriptor(device);
            bangDescriptor->createCnnlDescriptors();
            return (MatmulDescriptor *) (bangDescriptor);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            auto ascendDescriptor = new MatmulAclnnDescriptor(device);
            ascendDescriptor->createAclnnDescriptors();
            return (MatmulDescriptor *) (ascendDescriptor);
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C void destroyMatmulDescriptor(MatmulDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (MatmulCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (MatmulCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            auto bangDescriptor = (MatmulBangDescriptor *) (descriptor);
            bangDescriptor->destroyCnnlDescriptors();
            delete bangDescriptor;
            break;
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            auto ascendDescriptor = (MatmulAclnnDescriptor *) (descriptor);
            ascendDescriptor->destroyAclnnDescriptors();
            delete ascendDescriptor;
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

__C void matmul(MatmulDescriptor *descriptor, Tensor c, float beta, Tensor a, Tensor b, float alpha, void *stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            matmul_cpu_f16(c, beta, a, b, alpha);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            matmul_nv_gpu_f16(c, beta, a, b, alpha, stream);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            matmul_cnnl_f16((MatmulBangDescriptor *) (descriptor), c, beta, a, b, alpha, stream);
            break;
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu:
            matmul_aclnn_f16((MatmulAclnnDescriptor *) (descriptor), c, beta, a, b, alpha, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}
