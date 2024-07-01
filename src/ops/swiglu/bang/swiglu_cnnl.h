#ifndef __CNNL_SWIGLU_H__
#define __CNNL_SWIGLU_H__

#include "../../../operators.h"
#include "cnnl.h"
#include "cnnl_extra.h"

struct SwigluBangDescriptor {
    Device device;
    cnnlActivationDescriptor_t actDesc;
    cnnlBiasActivationGluDescriptor_t opDesc;

    SwigluBangDescriptor(Device device);
    void createCnnlDescriptors() {
        cnnlCreateActivationDescriptor(&actDesc);
        cnnlCreateBiasActivationGluDescriptor(&opDesc);
        cnnlSetActivationDescriptor_v6(actDesc, CNNL_ACTIVATION_SILU,
                                       CNNL_ACTIVATION_HIGH_PRECISION,
                                       CNNL_NOT_PROPAGATE_NAN,
                                       0.0, 0, 0.0, 0.0, true, true);
        cnnlSetBiasActivationGluDescriptor(opDesc, actDesc,
                                           CNNL_BIAS_ACTIVATION_GLU_ALGO_V2);
    }
    void destroyCnnlDescriptors() {
        cnnlDestroyActivationDescriptor(actDesc);
        cnnlDestroyBiasActivationGluDescriptor(opDesc);
    }
};

void swiglu_cnnl_f16(SwigluBangDescriptor *descriptor, Tensor gate, Tensor up, void *stream);

#endif// __CNNL_SWIGLU_H__
