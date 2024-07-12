#ifndef __CPU_SOFTMAX_H__
#define __CPU_SOFTMAX_H__

#include "../../../operators.h"
typedef struct SoftmaxCpuDescriptor {
    Device device;
} SoftmaxCpuDescriptor;

void softmax_cpu_f16(Tensor);

#endif
