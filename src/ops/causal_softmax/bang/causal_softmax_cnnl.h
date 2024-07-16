#ifndef __CNNL_CAUSAL_SOFTMAX_H__
#define __CNNL_CAUSAL_SOFTMAX_H__

#include "../../../operators.h"
#include "cnnl.h"
#include "cnnl_extra.h"

struct CausalSoftmaxBangDescriptor {
    Device device;
    cnnlTensorDescriptor_t tDesc;
    cnnlTensorDescriptor_t maskDesc;

    CausalSoftmaxBangDescriptor(Device device);
    void createCnnlDescriptors() {
        cnnlCreateTensorDescriptor(&tDesc);
        cnnlCreateTensorDescriptor(&maskDesc);
    }
    void destroyCnnlDescriptors() {
        cnnlDestroyTensorDescriptor(tDesc);
        cnnlDestroyTensorDescriptor(maskDesc);
    }
};

void causal_softmax_cnnl_f16(CausalSoftmaxBangDescriptor *descriptor, Tensor t, void *stream);

#endif// __CNNL_CAUSAL_SOFTMAX_H__
