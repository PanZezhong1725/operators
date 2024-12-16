#include "pooling.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include <numeric>

infiniopStatus_t cudaCreatePoolingDescriptor(CudaHandle_t handle,
                                             PoolingCudaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y,
                                             infiniopTensorDescriptor_t x,
                                             uint64_t const *kernel_shape,
                                             uint64_t const *pads,
                                             int64_t const *strides,
                                             uint64_t n,
                                             int pooling_type) {
    uint64_t ndim = y->ndim;
    if (ndim < 3 || ndim != x->ndim || ndim != n + 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (x->shape[0] != y->shape[0] || x->shape[1] != y->shape[1]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (pooling_type > 1) {
        return STATUS_BAD_PARAM;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    float alpha = 1.0f, beta = 0.0f;

    if (ndim <= 4) {

        int xn = x->shape[0];
        int xc = x->shape[1];
        int xh = ndim == 3 ? 1 : x->shape[2];
        int xw = ndim == 3 ? x->shape[2] : x->shape[3];
        int yh = ndim == 3 ? 1 : y->shape[2];
        int yw = ndim == 3 ? y->shape[2] : y->shape[3];
        const auto kernel_ = reinterpret_cast<int64_t const *>(kernel_shape);
        const auto pads_ = reinterpret_cast<int64_t const *>(pads);
        const auto strides_ = reinterpret_cast<int64_t const *>(strides);
        int kh = ndim == 3 ? 1 : kernel_[0];
        int kw = ndim == 3 ? kernel_[0] : kernel_[1];
        int ph = ndim == 3 ? 0 : pads_[0];
        int pw = ndim == 3 ? pads_[0] : pads_[1];
        int sh = ndim == 3 ? 1 : strides_[0];
        int sw = ndim == 3 ? strides_[0] : strides_[1];

        // get the data types of the tensors and the conv operator
        CREATE_CHECK_ERROR(auto tensor_dt = dataTypeMap[x->dt], tensor_dt, -1, STATUS_BAD_PARAM);

        // create and set tensor descriptors for x
        cudnnTensorDescriptor_t x_desc;
        checkCudnnError(cudnnCreateTensorDescriptor(&x_desc));
        checkCudnnError(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, static_cast<cudnnDataType_t>(tensor_dt), xn, xc, xh, xw));

        // Create and set pooling descriptor for average pooling
        cudnnPoolingDescriptor_t pool_desc;
        checkCudnnError(cudnnCreatePoolingDescriptor(&pool_desc));
        checkCudnnError(cudnnSetPooling2dDescriptor(pool_desc,
                                                    getPoolingMode(pooling_type),
                                                    CUDNN_NOT_PROPAGATE_NAN,
                                                    kh,// pooling window height
                                                    kw,// pooling window width
                                                    ph,// vertical padding
                                                    pw,// horizontal padding
                                                    sh,// vertical Stride
                                                    sw // horizontal stride
                                                    ));
        // create and set tensor descriptors for y
        cudnnTensorDescriptor_t y_desc;
        checkCudnnError(cudnnCreateTensorDescriptor(&y_desc));
        checkCudnnError(cudnnGetPooling2dForwardOutputDim(pool_desc, x_desc, &xn, &xc, &yh, &yw));
        checkCudnnError(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, static_cast<cudnnDataType_t>(tensor_dt), xn, xc, yh, yw));

        *desc_ptr = new PoolingCudaDescriptor{
            DevNvGpu,
            y->dt,
            handle->device_id,
            handle->cudnn_handles_t,
            x_desc,
            y_desc,
            pool_desc,
            alpha,
            beta,
        };
    } else {
        std::vector<int> x_shape(ndim);
        std::vector<int> x_strides(ndim);
        std::vector<int> y_shape(ndim);
        std::vector<int> y_strides(ndim);
        std::vector<int> k_shape(ndim - 2);
        std::vector<int> pads_int(ndim - 2);
        std::vector<int> strides_int(ndim - 2);

#pragma omp parallel for
        for (size_t i = 0; i < ndim; ++i) {
            x_shape[i] = static_cast<int>(x->shape[i]);
            x_strides[i] = static_cast<int>(x->strides[i]);
            y_shape[i] = static_cast<int>(y->shape[i]);
            y_strides[i] = static_cast<int>(y->strides[i]);
            if (i < ndim - 2) {
                k_shape[i] = static_cast<int>(kernel_shape[i]);
                pads_int[i] = static_cast<int>(pads[i]);
                strides_int[i] = static_cast<int>(strides[i]);
            }
        }

        // get the data types of the tensors and the conv operator
        CREATE_CHECK_ERROR(auto tensor_dt = dataTypeMap[x->dt], tensor_dt, -1, STATUS_BAD_PARAM);

        // create and set tensor descriptors for x
        cudnnTensorDescriptor_t x_desc;
        checkCudnnError(cudnnCreateTensorDescriptor(&x_desc));
        checkCudnnError(cudnnSetTensorNdDescriptor(x_desc, static_cast<cudnnDataType_t>(tensor_dt), ndim, x_shape.data(), x_strides.data()));

        // Create and set pooling descriptor for average pooling
        cudnnPoolingDescriptor_t pool_desc;
        checkCudnnError(cudnnCreatePoolingDescriptor(&pool_desc));
        checkCudnnError(cudnnSetPoolingNdDescriptor(pool_desc,
                                                    getPoolingMode(pooling_type),
                                                    CUDNN_NOT_PROPAGATE_NAN,
                                                    ndim - 2,
                                                    k_shape.data(),
                                                    pads_int.data(),
                                                    strides_int.data()));
        // create and set tensor descriptors for y
        cudnnTensorDescriptor_t y_desc;
        checkCudnnError(cudnnCreateTensorDescriptor(&y_desc));
        checkCudnnError(cudnnGetPoolingNdForwardOutputDim(pool_desc, x_desc, ndim, y_shape.data()));
        checkCudnnError(cudnnSetTensorNdDescriptor(y_desc, static_cast<cudnnDataType_t>(tensor_dt), ndim, y_shape.data(), y_strides.data()));

        *desc_ptr = new PoolingCudaDescriptor{
            DevNvGpu,
            y->dt,
            handle->device_id,
            handle->cudnn_handles_t,
            x_desc,
            y_desc,
            pool_desc,
            alpha,
            beta,
        };
        return STATUS_SUCCESS;
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGetPoolingWorkspaceSize(PoolingCudaDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyPoolingDescriptor(PoolingCudaDescriptor_t desc) {
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->x_desc));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->y_desc));
    checkCudnnError(cudnnDestroyPoolingDescriptor(desc->pool_desc));
    desc->cudnn_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
