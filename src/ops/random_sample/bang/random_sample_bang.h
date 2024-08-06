#ifndef __BANG_RANDOM_SAMPLE_H__
#define __BANG_RANDOM_SAMPLE_H__

#include "../../../operators.h"
#include "../../utils.h"
#include "cnrt.h"

struct RandomSampleBangDescriptor {
    Device device;
};

void random_sample_bang_f16(Tensor source, Tensor indices, Tensor index, float random, float topp, int topk, void *stream);

#endif// __BANG_RANDOM_SAMPLE_H__
