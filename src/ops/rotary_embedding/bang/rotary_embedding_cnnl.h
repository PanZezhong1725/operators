#ifndef __CNNL_ROTARY_EMBEDDING_H__
#define __CNNL_ROTARY_EMBEDDING_H__

#include "../../../operators.h"
#include "cnnl.h"
#include "cnnl_extra.h"

struct RotaryEmbeddingBangDescriptor {
    Device device;
    cnnlOpTensorDescriptor_t outerDesc;
    cnnlRotaryEmbeddingDescriptor_t ropeDesc;

    RotaryEmbeddingBangDescriptor(Device device);
    void createCnnlDescriptors() {
        cnnlCreateOpTensorDescriptor(&outerDesc);
        cnnlCreateRotaryEmbeddingDescriptor(&ropeDesc);
        cnnlSetOpTensorDescriptor(outerDesc, CNNL_OP_TENSOR_MUL,
                                  CNNL_DTYPE_FLOAT, CNNL_NOT_PROPAGATE_NAN);
        cnnlSetRotaryEmbeddingDescriptor_v2(ropeDesc, false, true,
                                            false, false, CNNL_SEQDATA_TNBC);
    }
    void destroyCnnlDescriptors() {
        cnnlDestroyOpTensorDescriptor(outerDesc);
        cnnlDestroyRotaryEmbeddingDescriptor(ropeDesc);
    }
};

void rotary_embedding_cnnl_f16(RotaryEmbeddingBangDescriptor *descriptor, Tensor t, Tensor pos, float theta, void *stream);

#endif// __CNNL_ROTARY_EMBEDDING_H__
