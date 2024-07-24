#ifndef __CNNL_SWIGLU_H__
#define __CNNL_SWIGLU_H__

#include "cnnl.h"
#include "cnnl_extra.h"
#include "operators.h"

struct SwigluBangDescriptor {
    Device device;
    cnnlTensorDescriptor_t gateDesc, inDesc;
    cnnlActivationDescriptor_t actDesc;
    cnnlBiasActivationGluDescriptor_t opDesc;

    SwigluBangDescriptor(Device device);
    void createCnnlDescriptors() {
        cnnlCreateTensorDescriptor(&gateDesc);
        cnnlCreateTensorDescriptor(&inDesc);
        cnnlCreateActivationDescriptor(&actDesc);
        cnnlCreateBiasActivationGluDescriptor(&opDesc);
    }
    void destroyCnnlDescriptors() {
        cnnlDestroyTensorDescriptor(gateDesc);
        cnnlDestroyTensorDescriptor(inDesc);
        cnnlDestroyActivationDescriptor(actDesc);
        cnnlDestroyBiasActivationGluDescriptor(opDesc);
    }
};

void swiglu_cnnl_f16(SwigluBangDescriptor *descriptor, Tensor gate, Tensor up, void *stream);

#endif// __CNNL_SWIGLU_H__
