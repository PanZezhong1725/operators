#ifndef __NV_GPU_SWIGLU_H__
#define __NV_GPU_SWIGLU_H__

#include "operators.h"

struct SwigluCudaDescriptor {
    Device device;
};

void swiglu_nv_gpu_f16(Tensor gate, Tensor up, void *stream);

#endif// __NV_GPU_SWIGLU_H__
