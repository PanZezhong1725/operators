#ifndef __CNNL_ROTARY_EMBEDDING_H__
#define __CNNL_ROTARY_EMBEDDING_H__

#include "cnnl.h"
#include "cnnl_extra.h"
#include "operators.h"

struct RotaryEmbeddingBangDescriptor {
    Device device;
    cnnlTensorDescriptor_t inDesc, posDesc, thetaDesc, freqDesc, freqConcatDesc, scalerDesc;
    cnnlOpTensorDescriptor_t outerDesc;
    cnnlRotaryEmbeddingDescriptor_t ropeDesc;

    RotaryEmbeddingBangDescriptor(Device device);
    void createCnnlDescriptors() {
        cnnlCreateTensorDescriptor(&inDesc);
        cnnlCreateTensorDescriptor(&posDesc);
        cnnlCreateTensorDescriptor(&thetaDesc);
        cnnlCreateTensorDescriptor(&freqDesc);
        cnnlCreateTensorDescriptor(&freqConcatDesc);
        cnnlCreateTensorDescriptor(&scalerDesc);
        cnnlCreateOpTensorDescriptor(&outerDesc);
        cnnlCreateRotaryEmbeddingDescriptor(&ropeDesc);
    }
    void destroyCnnlDescriptors() {
        cnnlDestroyTensorDescriptor(inDesc);
        cnnlDestroyTensorDescriptor(posDesc);
        cnnlDestroyTensorDescriptor(thetaDesc);
        cnnlDestroyTensorDescriptor(freqDesc);
        cnnlDestroyTensorDescriptor(freqConcatDesc);
        cnnlDestroyTensorDescriptor(scalerDesc);
        cnnlDestroyOpTensorDescriptor(outerDesc);
        cnnlDestroyRotaryEmbeddingDescriptor(ropeDesc);
    }
};

void rotary_embedding_cnnl_f16(RotaryEmbeddingBangDescriptor *descriptor, Tensor t, Tensor pos, float theta, void *stream);

#endif// __CNNL_ROTARY_EMBEDDING_H__
