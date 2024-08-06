#ifndef __BANG_CAUSAL_SOFTMAX_H__
#define __BANG_CAUSAL_SOFTMAX_H__

#include "../../utils.h"
#include "cnrt.h"
#include "operators.h"

void causal_softmax_bang_f16(Tensor y, void *stream);

#endif// __BANG_CAUSAL_SOFTMAX_H__

