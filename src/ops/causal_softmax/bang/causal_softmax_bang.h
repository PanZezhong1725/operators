#ifndef __BANG_MASK_SOFTMAX_H__
#define __BANG_MASK_SOFTMAX_H__

#include "../../../operators.h"
#include "../../utils.h"
#include "cnrt.h"

void maskSoftmax_bang_f16(Tensor y, Tensor x, void *stream);

#endif// __BANG_MASK_SOFTMAX_H__

