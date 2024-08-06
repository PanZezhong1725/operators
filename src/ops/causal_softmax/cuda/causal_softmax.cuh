#ifndef __NV_CPU_CAUSAL_SOFTMAX_H__
#define __NV_CPU_CAUSAL_SOFTMAX_H__

#include "operators.h"

typedef struct CausalSoftmaxCudaDescriptor {
    Device device;
} CausalSoftmaxCudaDescriptor;

void causal_softmax_nv_gpu_f16(CausalSoftmaxCudaDescriptor *, Tensor, void *stream);

#endif
