#ifndef __TOPK_H__
#define __TOPK_H__

#include "../../export.h"
#include "../../operators.h"

typedef struct TopKDescriptor TopKDescriptor;

__C __export void *createTopKDescriptor(Device, void *config);
__C __export void destroyTopKDescriptor(TopKDescriptor *descriptor);
__C __export void topK(TopKDescriptor *descriptor, Tensor indices, Tensor probs, Tensor logits, int64_t k, void *stream);

#endif
