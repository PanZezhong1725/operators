#ifndef __CPU_SWIGLU_H__
#define __CPU_SWIGLU_H__

#include "../../../operators.h"

typedef struct SwigluCpuDescriptor {
    Device device;
} SwigluCpuDescriptor;

void swiglu_cpu_f16(Tensor gate, Tensor up);

#endif// __CPU_SWIGLU_H__
