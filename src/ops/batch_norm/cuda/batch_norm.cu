#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "batch_norm.cuh"

infiniopStatus_t batch_norm_nv_gpu(BatchNormCudaDescriptor_t desc, void *y, void const *x, void const *scale, void const *b, void const *mean, void const *var, void *stream) {
    checkCudnnError(use_cudnn(desc->cudnn_handles_t, desc->device_id, (cudaStream_t) stream,
                              [&](cudnnHandle_t handle) { return cudnnBatchNormalizationForwardInference(handle, desc->mode, &desc->alpha, &desc->beta,
                                                                                                         desc->x_desc, x, desc->y_desc, y, desc->bn_desc,
                                                                                                         scale, b, mean, var, desc->eps); }));
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaBatchNorm(BatchNormCudaDescriptor_t desc, void *y, void const *x,
                               void const *scale, void const *b, void const *mean, void const *var,
                               void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return batch_norm_nv_gpu(desc, y, x, scale, b, mean, var, stream);
    }
    if (desc->dtype == F32) {
        return batch_norm_nv_gpu(desc, y, x, scale, b, mean, var, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
