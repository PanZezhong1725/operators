#ifndef __CNNL_CAUSAL_SOFTMAX_H__
#define __CNNL_CAUSAL_SOFTMAX_H__

#include "cnnl.h"
#include "cnnl_extra.h"
#include "operators.h"

struct CausalSoftmaxBangDescriptor {
    Device device;
    CausalSoftmaxBangDescriptor(Device device);
};

void causal_softmax_cnnl_f16(Tensor t, void *stream);

#endif// __CNNL_CAUSAL_SOFTMAX_H__
