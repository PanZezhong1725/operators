#ifndef __CNNL_RMS_NORM_H__
#define __CNNL_RMS_NORM_H__

#include "cnnl.h"
#include "cnnl_extra.h"
#include "operators.h"

struct RMSNormBangDescriptor {
    Device device;
    cnnlTensorDescriptor_t yDesc, xDesc, wDesc;
    cnnlFuseNormDescriptor_t opDesc;

    RMSNormBangDescriptor(Device device);
    void createCnnlDescriptors() {
        cnnlCreateTensorDescriptor(&yDesc);
        cnnlCreateTensorDescriptor(&xDesc);
        cnnlCreateTensorDescriptor(&wDesc);
        cnnlCreateFuseNormDescriptor(&opDesc);
    }
    void destroyCnnlDescriptors() {
        cnnlDestroyFuseNormDescriptor(opDesc);
        cnnlDestroyTensorDescriptor(xDesc);
        cnnlDestroyTensorDescriptor(yDesc);
        cnnlDestroyTensorDescriptor(wDesc);
    }
};

void rms_norm_cnnl_f16(RMSNormBangDescriptor *descriptor, Tensor y, Tensor x, Tensor w, float epsilon, void *stream);

#endif// __CNNL_RMS_NORM_H__
