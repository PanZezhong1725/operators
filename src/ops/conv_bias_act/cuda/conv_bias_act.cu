#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "conv_bias_act.cuh"

infiniopStatus_t conv_bias_act_nv_gpu(ConvBiasActCudaDescriptor_t desc, void *workspace, uint64_t workspace_size,
                                      void *y, void const *x, void const *w, void const *b, void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    checkCudnnError(use_cudnn(desc->cudnn_handles_t, desc->device_id, (cudaStream_t) stream,
                              [&](cudnnHandle_t handle) { return cudnnConvolutionBiasActivationForward(handle, &desc->alpha,
                                                                                                       desc->x_desc, x, desc->w_desc, w, desc->op_desc, desc->algo, workspace, workspace_size,
                                                                                                       &desc->beta, desc->y_desc, y, desc->b_desc, b, desc->act_desc, desc->y_desc, y); }));
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaConvBiasAct(ConvBiasActCudaDescriptor_t desc,
                                 void *workspace, uint64_t workspace_size,
                                 void *y, void const *x, void const *w,
                                 void const *b, void *stream) {
    if (desc->dtype == F16 || desc->dtype == F32) {
        return conv_bias_act_nv_gpu(desc, workspace, workspace_size, y, x, w, b, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
