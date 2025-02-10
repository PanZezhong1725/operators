#include "batch_norm.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateBatchNormDescriptor(CudaHandle_t handle,
                                               BatchNormCudaDescriptor_t *desc_ptr,
                                               infiniopTensorDescriptor_t y,
                                               infiniopTensorDescriptor_t x,
                                               infiniopTensorDescriptor_t scale,
                                               infiniopTensorDescriptor_t b,
                                               infiniopTensorDescriptor_t mean,
                                               infiniopTensorDescriptor_t var,
                                               double eps) {
    uint64_t ndim = y->ndim;
    if (ndim != x->ndim || scale->ndim != b->ndim || scale->ndim != mean->ndim || scale->ndim != var->ndim || scale->ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (y->shape[i] != x->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (x->shape[1] != scale->shape[0] || scale->shape[0] != b->shape[0] || scale->shape[0] != mean->shape[0] || scale->shape[0] != var->shape[0]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (eps < CUDNN_BN_MIN_EPSILON) {
        return STATUS_BAD_PARAM;
    }

    const auto new_ndim = std::max(4UL, ndim);
    int32_t x_shape[new_ndim];
    int32_t y_shape[new_ndim];
    int32_t x_strides[new_ndim];
    int32_t y_strides[new_ndim];
    int32_t bn_shape[new_ndim];
    int32_t bn_strides[new_ndim];
    for (size_t i = 0; i < new_ndim; ++i) {
        x_shape[i] = i < ndim ? static_cast<int32_t>(x->shape[i]) : 1;
        x_strides[i] = i < ndim ? static_cast<int32_t>(x->strides[i]) : 1;
        y_shape[i] = i < ndim ? static_cast<int32_t>(y->shape[i]) : 1;
        y_strides[i] = i < ndim ? static_cast<int32_t>(y->strides[i]) : 1;
        bn_shape[i] = i == 1 ? x->shape[i] : 1;
        bn_strides[i] = 1;
    }

    // get the data types of the tensors and the conv operator
    CREATE_CHECK_ERROR(auto tensor_dt = dataTypeMap[x->dt], tensor_dt, -1, STATUS_BAD_PARAM);
    cudnnDataType_t bn_dt = [&] {
        switch (tensor_dt) {
            case CUDNN_DATA_INT8:
            case CUDNN_DATA_HALF:
            case CUDNN_DATA_FLOAT:
                return CUDNN_DATA_FLOAT;
            default:
                return CUDNN_DATA_DOUBLE;
        }
    }();

    // get the input tensor descriptor
    cudnnTensorDescriptor_t x_desc;
    checkCudnnError(cudnnCreateTensorDescriptor(&x_desc));
    checkCudnnError(cudnnSetTensorNdDescriptor(x_desc, static_cast<cudnnDataType_t>(tensor_dt), new_ndim, x_shape, x_strides));

    // get the secondary tensor descriptor
    cudnnTensorDescriptor_t bn_desc;
    cudnnBatchNormMode_t mode;
    checkCudnnError(cudnnCreateTensorDescriptor(&bn_desc));
    if (handle->compute_capability_major > 6 || (handle->compute_capability_major == 6 && handle->compute_capability_minor >= 0)) {
        mode = CUDNN_BATCHNORM_SPATIAL;
    } else {
        mode = CUDNN_BATCHNORM_SPATIAL;
    }
    // checkCudnnError(cudnnDeriveBNTensorDescriptor(bn_desc, x_desc, mode));
    checkCudnnError(cudnnSetTensorNdDescriptor(bn_desc, static_cast<cudnnDataType_t>(bn_dt), new_ndim, bn_shape, bn_strides));

    // get the output tensor descriptor
    cudnnTensorDescriptor_t y_desc;
    checkCudnnError(cudnnCreateTensorDescriptor(&y_desc));
    checkCudnnError(cudnnSetTensorNdDescriptor(y_desc, static_cast<cudnnDataType_t>(tensor_dt), new_ndim, y_shape, y_strides));

    float alpha = 1.0f, beta = 0.0f;

    *desc_ptr = new BatchNormCudaDescriptor{
        DevNvGpu,
        y->dt,
        handle->device_id,
        handle->cudnn_handles_t,
        x_desc,
        bn_desc,
        y_desc,
        alpha,
        beta,
        eps,
        mode,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyBatchNormDescriptor(BatchNormCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
