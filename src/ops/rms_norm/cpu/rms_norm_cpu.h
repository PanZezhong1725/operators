#ifndef __CPU_RMS_NORM_H__
#define __CPU_RMS_NORM_H__

#include "operators.h"

struct RMSNormCpuDescriptor {
    Device device;
};

void rms_norm_cpu_f16(Tensor y, Tensor x, Tensor w, float epsilon);

#endif// __CPU_RMS_NORM_H__
