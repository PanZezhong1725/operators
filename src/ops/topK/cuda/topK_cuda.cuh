#ifndef __NV_GPU_TOPK_H__
#define __NV_GPU_TOPK_H__

#include "../../../operators.h"

typedef struct TopKCudaDescriptor {
    Device device;
} TopKCudaDescriptor;

void topK_nv_gpu_f16(Tensor, Tensor, Tensor, int64_t, void *stream);

#endif
