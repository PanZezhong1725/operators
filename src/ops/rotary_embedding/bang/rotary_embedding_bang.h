#ifndef __BANG_ROTARY_EMBEDDING_H__
#define __BANG_ROTARY_EMBEDDING_H__

#include "../../../operators.h"
#include "../../utils.h"
#include "cnrt.h"

void rotaryEmbedding_bang_f16(Tensor t, Tensor pos, float theta, void *stream);

#endif// __BANG_ROTARY_EMBEDDING_H__
