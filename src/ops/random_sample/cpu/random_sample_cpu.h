#ifndef __CPU_RANDOM_SAMPLE_H__
#define __CPU_RANDOM_SAMPLE_H__

#include "../../../operators.h"

struct RandomSampleCpuDescriptor {
    Device device;
};

void random_sample_cpu_f16(Tensor source, Tensor indices, Tensor index, float random, float topp, int topk);

#endif// __CPU_RANDOM_SAMPLE_H__
