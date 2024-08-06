#ifndef __CPU_REFORM_H__
#define __CPU_REFORM_H__

#include "operators.h"

struct ReformCpuDescriptor {
    Device device;
};

void reform_cpu(Tensor y, Tensor x);

#endif// __CPU_REFORM_H__
