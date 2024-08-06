#ifndef __NV_GPU_RANDOM_SAMPLE_H__
#define __NV_GPU_RANDOM_SAMPLE_H__

#include "../../../operators.h"

struct RandomSampleCudaDescriptor {
    Device device;
};

void random_sample_nv_gpu_f16(Tensor source, Tensor indices, Tensor index, float random, float topp, int topk, void *stream);

#endif// __NV_GPU_RANDOM_SAMPLE_H__
