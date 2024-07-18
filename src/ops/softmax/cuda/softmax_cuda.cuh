#ifndef __NV_CPU_SOFTMAX_H__
#define __NV_CPU_SOFTMAX_H__

#include "../../../operators.h"

typedef struct SoftmaxCudaDescriptor {
    Device device;
} SoftmaxCudaDescriptor;

void softmax_nv_gpu_f16(SoftmaxCudaDescriptor *, Tensor, void *stream);

#endif
