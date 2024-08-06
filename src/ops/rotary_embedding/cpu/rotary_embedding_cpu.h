#ifndef __CPU_ROTARY_EMBEDDING_H__
#define __CPU_ROTARY_EMBEDDING_H__

#include "operators.h"

struct RotaryEmbeddingCpuDescriptor {
    Device device;
};

void rotary_embedding_cpu_f16(Tensor t, Tensor pos, float theta);

#endif// __CPU_RMS_NORM_H__
