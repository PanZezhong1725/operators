#ifndef __CPU_CAUSAL_SOFTMAX_H__
#define __CPU_CAUSAL_SOFTMAX_H__

#include "operators.h"
typedef struct CausalSoftmaxCpuDescriptor {
    Device device;
} CausalSoftmaxCpuDescriptor;

void causal_softmax_cpu_f16(Tensor);

#endif
