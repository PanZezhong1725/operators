#ifndef __NV_GPU_ROTARY_EMBEDDING_H__
#define __NV_GPU_ROTARY_EMBEDDING_H__

#include "operators.h"

struct RotaryEmbeddingCudaDescriptor {
    Device device;
};

void rotary_embedding_nv_gpu_f16(Tensor t, Tensor pos, float theta, void *stream);

#endif// __NV_GPU_ROTARY_EMBEDDING_H__
