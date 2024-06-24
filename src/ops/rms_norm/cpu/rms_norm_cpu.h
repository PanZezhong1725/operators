﻿#ifndef __CPU_RMS_NORM_H__
#define __CPU_RMS_NORM_H__

#include "../../../operators.h"

typedef struct RMSNormCpuDescriptor {
    Device device;
} RMSNormCpuDescriptor;

void rms_norm_cpu_f16(Tensor y, Tensor x, Tensor w, float epsilon);

#endif// __CPU_RMS_NORM_H__
