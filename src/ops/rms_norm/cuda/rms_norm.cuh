﻿#ifndef __NV_GPU_RMS_NORM_H__
#define __NV_GPU_RMS_NORM_H__

#include "../../../operators.h"

void rms_norm_nv_gpu_f16(MutTensor y, ConstTensor x, ConstTensor w, float epsilon, void *stream);

#endif// __NV_GPU_RMS_NORM_H__
