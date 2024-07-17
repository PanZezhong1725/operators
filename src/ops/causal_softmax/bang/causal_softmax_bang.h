#ifndef __BANG_CAUSAL_SOFTMAX_H__
#define __BANG_CAUSAL_SOFTMAX_H__

#include "../../../operators.h"
#include "../../utils.h"
#include "cnrt.h"

void causal_softmax_bang_f16(Tensor y, Tensor x, void *stream);

#endif// __BANG_CAUSAL_SOFTMAX_H__

