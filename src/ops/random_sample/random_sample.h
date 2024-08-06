#ifndef RANDOM_SAMPLE_H
#define RANDOM_SAMPLE_H

#include "../../export.h"
#include "../../operators.h"

typedef struct RandomSampleDescriptor RandomSampleDescriptor;

__C __export void *createRandomSampleDescriptor(Device, void *config);
__C __export void destroyRandomSampleDescriptor(RandomSampleDescriptor *descriptor);
__C __export void random_sample(RandomSampleDescriptor *descriptor, Tensor source, Tensor indices, Tensor index, float random, float topp, int topk, void *stream);

#endif
