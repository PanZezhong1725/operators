#ifndef __NV_GPU_REFORM_H__
#define __NV_GPU_REFORM_H__

#include "../../../devices/cuda/common_cuda.h"
#include "operators.h"

struct ReformCudaDescriptor {
    Device device;
};

void reform_nv_gpu(Tensor y, Tensor x, void *stream);

#endif// __NV_GPU_REFORM_H__
