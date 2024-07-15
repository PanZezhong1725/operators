#ifndef __BANG_SWIGLU_H__
#define __BANG_SWIGLU_H__

#include "../../../operators.h"
#include "../../utils.h"
#include "cnrt.h"

void swiglu_bang_f16(Tensor gate, Tensor up, void *stream);

#endif// __BANG_SWIGLU_H__
