#include "rotary_embedding_cnnl.h"
#include "../../../devices/bang/common_bang.h"
#include "../../../devices/bang/handle_pool.h"
#include "../../utils.h"
#include "cnrt.h"

RotaryEmbeddingBangDescriptor::RotaryEmbeddingBangDescriptor(Device device) {
    this->device = device;
    get_cnnl_pool();
}

void rotary_embedding_cnnl_f16(RotaryEmbeddingBangDescriptor *descriptor, Tensor t, Tensor pos, float theta, void *stream) {
    ASSERT_EQ(t.layout->ndim, 3);
    ASSERT_EQ(pos.layout->ndim, 1);
    ASSERT_EQ(pos.layout->shape[0], t.layout->shape[0]);

    auto nt = static_cast<int>(t.layout->shape[0]),
         nh = static_cast<int>(t.layout->shape[1]),
         dh = static_cast<int>(t.layout->shape[2]);

    int inDim[4] = {nt, 1, nh, dh};
    int posDim[2] = {nt, 1};
    int thetaDim[2] = {1, dh / 2};
    int freqDim[2] = {nt, dh / 2};
    int freqConcatDim[2] = {nt, dh};
    int scalerDim[1] = {1};

    cnnlTensorDescriptor_t inDesc, posDesc, thetaDesc, freqDesc, freqConcatDesc, scalerDesc;
    cnnlCreateTensorDescriptor(&inDesc);
    cnnlCreateTensorDescriptor(&posDesc);
    cnnlCreateTensorDescriptor(&thetaDesc);
    cnnlCreateTensorDescriptor(&freqDesc);
    cnnlCreateTensorDescriptor(&freqConcatDesc);
    cnnlCreateTensorDescriptor(&scalerDesc);

    cnnlSetTensorDescriptor(inDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF, 4, inDim);
    cnnlSetTensorDescriptor(posDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32, 2, posDim);
    cnnlSetTensorDescriptor(thetaDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, thetaDim);
    cnnlSetTensorDescriptor(freqDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, freqDim);
    cnnlSetTensorDescriptor(freqConcatDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, freqConcatDim);
    cnnlSetTensorDescriptor(scalerDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, scalerDim);

    void *thetaData, *freqData, *freqConcatData, *scalerData;
    cnrtMalloc(&thetaData, dh / 2 * sizeof(float));
    cnrtMalloc(&freqData, nt * dh / 2 * sizeof(float));
    cnrtMalloc(&freqConcatData, nt * dh * sizeof(float));
    cnrtMalloc(&scalerData, sizeof(float));

    void *powWorkspace, *outerWorkspace, *concatWorkspace;
    float zero = 0.0f, one = 1.0f;
    float scaler = -2.0f / dh;

    use_cnnl((cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 cnrtMemcpy(scalerData, &scaler, sizeof(float), cnrtMemcpyHostToDev);
                 // Use Arange to get [0, 1, 2, ..., dh / 2]
                 cnnlArange_v2(handle, CNNL_COMPUTATION_ULTRAHIGH_PRECISION, &zero,
                               &scaler, thetaDesc, thetaData);

                 // Use PowR to calc ((theta)^(-2/d))^n
                 cnrtMemcpy(scalerData, &theta, sizeof(float), cnrtMemcpyHostToDev);

                 size_t powWorkspaceSize;
                 cnnlGetPowWorkspaceSize(handle, scalerDesc, thetaDesc,
                                         thetaDesc, &powWorkspaceSize);
                 cnrtMalloc(&powWorkspace, powWorkspaceSize);

                 cnnlPow(handle, CNNL_COMPUTATION_ULTRAHIGH_PRECISION,
                         scalerDesc, scalerData, thetaDesc, thetaData,
                         powWorkspace, powWorkspaceSize, thetaDesc, thetaData);

                 // Use Broadcast Mul to calc t * theta_n
                 size_t outerWorkspaceSize;
                 cnnlGetOpTensorWorkspaceSize_v2(handle, descriptor->outerDesc, &one,
                                                 posDesc, pos.data,
                                                 &one, thetaDesc, thetaData,
                                                 &zero, freqDesc, freqData,
                                                 &outerWorkspaceSize);
                 cnrtMalloc(&outerWorkspace, outerWorkspaceSize);

                 cnnlOpTensor(handle, descriptor->outerDesc, &one,
                              posDesc, pos.data,
                              &one, thetaDesc, thetaData,
                              outerWorkspace, outerWorkspaceSize,
                              &zero, freqDesc, freqData);

                 // Concat two freqs to get [freq, freq]
                 size_t concatWorkspaceSize;
                 cnnlGetConcatWorkspaceSize(handle, 2, &concatWorkspaceSize);
                 cnrtMalloc(&concatWorkspace, concatWorkspaceSize);

                 cnnlTensorDescriptor_t concatDescs[2] = {freqDesc, freqDesc};
                 void *const concatData[2] = {freqData, freqData};

                 cnnlConcat(handle, 2, -1, concatDescs, concatData,
                            concatWorkspace, concatWorkspaceSize,
                            freqConcatDesc, freqConcatData);

                 // Do RotaryEmbedding with t(fp16) and [freq, freq](fp32)
                 cnnlRotaryEmbedding_v2(handle, descriptor->ropeDesc, inDesc, t.data,
                                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        freqConcatDesc, freqConcatData, 
                                        nullptr, nullptr, nullptr, 0,
                                        inDesc, t.data, nullptr, nullptr);
             });

    cnrtFree(thetaData);
    cnrtFree(freqData);
    cnrtFree(freqConcatData);
    cnrtFree(scalerData);
    cnrtFree(powWorkspace);
    cnrtFree(outerWorkspace);
    cnrtFree(concatWorkspace);

    cnnlDestroyTensorDescriptor(inDesc);
    cnnlDestroyTensorDescriptor(posDesc);
    cnnlDestroyTensorDescriptor(thetaDesc);
    cnnlDestroyTensorDescriptor(freqDesc);
    cnnlDestroyTensorDescriptor(freqConcatDesc);
    cnnlDestroyTensorDescriptor(scalerDesc);
}
