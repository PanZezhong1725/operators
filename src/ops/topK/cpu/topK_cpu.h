#ifndef __TOPK_CPU_H__
#define __TOPK_CPU_H__

#include "../../../operators.h"
typedef struct TopKCpuDescriptor {
    Device device;
} TopKCpuDescriptor;

void topK_cpu(Tensor indices, Tensor probs, Tensor logits, int64_t k);

#endif
