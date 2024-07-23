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
    int inDimStride[4] = {static_cast<int>(t.layout->strides[0] / t.layout->dt.size),
                          0,
                          static_cast<int>(t.layout->strides[1] / t.layout->dt.size),
                          static_cast<int>(t.layout->strides[2] / t.layout->dt.size)};
    int posDim[2] = {nt, 1};
    int thetaDim[2] = {1, dh / 2};
    int freqDim[2] = {nt, dh / 2};
    int freqConcatDim[2] = {nt, dh};
    int scalerDim[1] = {1};

    cnnlSetTensorDescriptor(descriptor->posDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32, 2, posDim);
    cnnlSetTensorDescriptorEx(descriptor->inDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF, 4, inDim, inDimStride);
    cnnlSetTensorDescriptor(descriptor->thetaDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, thetaDim);
    cnnlSetTensorDescriptor(descriptor->freqDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, freqDim);
    cnnlSetTensorDescriptor(descriptor->freqConcatDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, freqConcatDim);
    cnnlSetTensorDescriptor(descriptor->scalerDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, scalerDim);

    void *thetaData, *freqData, *freqConcatData, *scalerData;
    cnrtMalloc(&thetaData, dh / 2 * sizeof(float) + nt * dh / 2 * sizeof(float) + nt * dh * sizeof(float) + sizeof(float));
    freqData = static_cast<char *>(thetaData) + dh / 2 * sizeof(float);
    freqConcatData = static_cast<char *>(freqData) + nt * dh / 2 * sizeof(float);
    scalerData = static_cast<char *>(freqConcatData) + nt * dh * sizeof(float);

    cnnlSetOpTensorDescriptor(descriptor->outerDesc, CNNL_OP_TENSOR_MUL,
                              CNNL_DTYPE_FLOAT, CNNL_NOT_PROPAGATE_NAN);
    cnnlSetRotaryEmbeddingDescriptor_v2(descriptor->ropeDesc, false, true,
                                        false, false, CNNL_SEQDATA_TNBC);

    void *powWorkspace, *outerWorkspace, *concatWorkspace;
    float zero = 0.0f, one = 1.0f;
    float scaler = -2.0f / dh;

    use_cnnl((cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 cnrtMemcpy(scalerData, &scaler, sizeof(float), cnrtMemcpyHostToDev);

                 void *workspace;
                 size_t workspaceSize = 0;
                 size_t powWorkspaceSize;
                 cnnlGetPowWorkspaceSize(handle, descriptor->scalerDesc, descriptor->thetaDesc,
                                         descriptor->thetaDesc, &powWorkspaceSize);
                 workspaceSize += powWorkspaceSize;

                 // Use Broadcast Mul to calc t * theta_n
                 size_t outerWorkspaceSize;
                 cnnlGetOpTensorWorkspaceSize_v2(handle, descriptor->outerDesc, &one,
                                                 descriptor->posDesc, pos.data,
                                                 &one, descriptor->thetaDesc, thetaData,
                                                 &zero, descriptor->freqDesc, freqData,
                                                 &outerWorkspaceSize);
                 workspaceSize += outerWorkspaceSize;

                 // Concat two freqs to get [freq, freq]
                 size_t concatWorkspaceSize;
                 cnnlGetConcatWorkspaceSize(handle, 2, &concatWorkspaceSize);
                 workspaceSize += concatWorkspaceSize;

                 cnrtMalloc(&workspace, workspaceSize);
                 powWorkspace = workspace;
                 outerWorkspace = static_cast<char *>(powWorkspace) + powWorkspaceSize;
                 concatWorkspace = static_cast<char *>(outerWorkspace) + outerWorkspaceSize;

                 // Use Arange to get [0, 1, 2, ..., dh / 2]
                 cnnlArange_v2(handle, CNNL_COMPUTATION_ULTRAHIGH_PRECISION, &zero,
                               &scaler, descriptor->thetaDesc, thetaData);

                 // Use PowR to calc ((theta)^(-2/d))^n
                 cnrtMemcpy(scalerData, &theta, sizeof(float), cnrtMemcpyHostToDev);


                 cnnlPow(handle, CNNL_COMPUTATION_ULTRAHIGH_PRECISION,
                         descriptor->scalerDesc, scalerData, descriptor->thetaDesc, thetaData,
                         powWorkspace, powWorkspaceSize, descriptor->thetaDesc, thetaData);


                 cnnlOpTensor(handle, descriptor->outerDesc, &one,
                              descriptor->posDesc, pos.data,
                              &one, descriptor->thetaDesc, thetaData,
                              outerWorkspace, outerWorkspaceSize,
                              &zero, descriptor->freqDesc, freqData);


                 cnnlTensorDescriptor_t concatDescs[2] = {descriptor->freqDesc, descriptor->freqDesc};
                 void *const concatData[2] = {freqData, freqData};

                 cnnlConcat(handle, 2, -1, concatDescs, concatData,
                            concatWorkspace, concatWorkspaceSize,
                            descriptor->freqConcatDesc, freqConcatData);

                 // Do RotaryEmbedding with t(fp16) and [freq, freq](fp32)
                 cnnlRotaryEmbedding_v2(handle, descriptor->ropeDesc, descriptor->inDesc, t.data,
                                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        descriptor->freqConcatDesc, freqConcatData,
                                        nullptr, nullptr, nullptr, 0,
                                        descriptor->inDesc, t.data, nullptr, nullptr);
             });

    cnrtFree(thetaData);
    cnrtFree(powWorkspace);
}
