#ifndef __CNNL_RMS_NORM_H__
#define __CNNL_RMS_NORM_H__

#include "cnnl.h"
#include "cnnl_extra.h"
#include "operators.h"

struct RMSNormCnnlDescriptor {
    Device device;
    RMSNormCnnlDescriptor(Device device);
};

void rms_norm_cnnl_f16(Tensor y, Tensor x, Tensor w, float epsilon, void *stream);

#endif// __CNNL_RMS_NORM_H__
