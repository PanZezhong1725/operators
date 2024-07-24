#ifndef __BANG_RMS_NORM_H__
#define __BANG_RMS_NORM_H__

#include "../../utils.h"
#include "cnrt.h"
#include "operators.h"

void rms_norm_bang_f16(Tensor y, Tensor x, Tensor w, float epsilon, void *stream);

#endif// __BANG_RMS_NORM_H__
