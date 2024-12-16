#include "global_avg_pool.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateGlobalAvgPoolDescriptor(CudaHandle_t handle,
                                                   GlobalAvgPoolCudaDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y,
                                                   infiniopTensorDescriptor_t x) {
    uint64_t ndim = y->ndim;
    if (ndim <= 2 || ndim != x->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (i < 2 && y->shape[i] != x->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        } else if (i >= 2 && y->shape[i] != 1) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
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

    // use cuDNN lib call
    if (x->ndim <= 4) {
        int n = x->shape[0];
        int c = x->shape[1];
        int h = ndim == 3 ? 1 : x->shape[2];
        int w = ndim == 3 ? x->shape[2] : x->shape[3];

        // get the data types of the tensors and the conv operator
        CREATE_CHECK_ERROR(auto tensor_dt = dataTypeMap[x->dt], tensor_dt, -1, STATUS_BAD_PARAM);

        // create and set tensor descriptor for x
        cudnnTensorDescriptor_t x_desc;
        checkCudnnError(cudnnCreateTensorDescriptor(&x_desc));
        checkCudnnError(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, static_cast<cudnnDataType_t>(tensor_dt), n, c, h, w));

        // create and set tensor descriptor for y
        cudnnTensorDescriptor_t y_desc;
        checkCudnnError(cudnnCreateTensorDescriptor(&y_desc));
        checkCudnnError(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, static_cast<cudnnDataType_t>(tensor_dt), n, c, 1, 1));

        // Create and set pooling descriptor for average pooling
        cudnnPoolingDescriptor_t pool_desc;
        checkCudnnError(cudnnCreatePoolingDescriptor(&pool_desc));
        checkCudnnError(cudnnSetPooling2dDescriptor(pool_desc,
                                                    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                                    CUDNN_NOT_PROPAGATE_NAN,
                                                    h,// pooling window height
                                                    w,// pooling window width
                                                    0,// vertical padding
                                                    0,// horizontal padding
                                                    1,// vertical Stride
                                                    1 // horizontal stride
                                                    ));
        float alpha = 1.0f, beta = 0.0f;

        *desc_ptr = new GlobalAvgPoolCudaDescriptor{
            DevNvGpu,
            y->dt,
            handle->device_id,
            ndim,
            0,
            0,
            0,
            0,
            0,
            0,
            handle->cudnn_handles_t,
            x_desc,
            y_desc,
            pool_desc,
            alpha,
            beta,
        };

    } else if (x->ndim <= 5) {
        std::vector<int> x_shape(ndim);
        std::vector<int> x_strides(ndim);
        std::vector<int> y_shape(ndim);
        std::vector<int> y_strides(ndim);
        std::vector<int> k_shape(ndim - 2);
        std::vector<int> pads(ndim - 2);
        std::vector<int> strides(ndim - 2);

#pragma omp parallel for
        for (size_t i = 0; i < ndim; ++i) {
            x_shape[i] = static_cast<int>(x->shape[i]);
            x_strides[i] = static_cast<int>(x->strides[i]);
            y_shape[i] = static_cast<int>(y->shape[i]);
            y_strides[i] = static_cast<int>(y->strides[i]);
            if (i < ndim - 2) {
                k_shape[i] = static_cast<int>(x->shape[i + 2]);
                pads[i] = 0;
                strides[i] = 1;
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
                                                    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                                    CUDNN_NOT_PROPAGATE_NAN,
                                                    ndim - 2,
                                                    k_shape.data(),
                                                    pads.data(),
                                                    strides.data()));
        // create and set tensor descriptors for y
        cudnnTensorDescriptor_t y_desc;
        checkCudnnError(cudnnCreateTensorDescriptor(&y_desc));
        checkCudnnError(cudnnGetPoolingNdForwardOutputDim(pool_desc, x_desc, ndim, y_shape.data()));
        checkCudnnError(cudnnSetTensorNdDescriptor(y_desc, static_cast<cudnnDataType_t>(tensor_dt), ndim, y_shape.data(), y_strides.data()));

        float alpha = 1.0f, beta = 0.0f;

        *desc_ptr = new GlobalAvgPoolCudaDescriptor{
            DevNvGpu,
            y->dt,
            handle->device_id,
            ndim,
            0,
            0,
            0,
            0,
            0,
            0,
            handle->cudnn_handles_t,
            x_desc,
            y_desc,
            pool_desc,
            alpha,
            beta,
        };

    } else {
        uint64_t y_data_size = std::accumulate(y->shape, y->shape + 2, 1ULL, std::multiplies<uint64_t>());
        uint64_t x_per_NC_data_size = std::accumulate(x->shape + 2, x->shape + ndim, 1ULL, std::multiplies<uint64_t>());
        uint64_t data_size = y_data_size * x_per_NC_data_size;

        unsigned max_block_size = std::min(256, handle->prop.maxThreadsPerBlock);
        uint64_t max_grid_size = static_cast<uint64_t>(handle->prop.maxGridSize[0]);
        uint64_t items_per_thread = data_size / (max_block_size * max_grid_size);

        *desc_ptr = new GlobalAvgPoolCudaDescriptor{
            DevNvGpu,
            y->dt,
            handle->device_id,
            ndim,
            data_size,
            y_data_size,
            x_per_NC_data_size,
            max_block_size,
            max_grid_size,
            items_per_thread,
            handle->cudnn_handles_t,
            nullptr,
            nullptr,
            nullptr,
            0,
            0,
        };
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGetGlobalAvgPoolWorkspaceSize(GlobalAvgPoolCudaDescriptor_t desc, uint64_t *size) {
    *size = desc->ndim <= 5 ? 0 : (desc->dtype != F16 ? 0 : std::min(desc->dtype.size * 2, 8) * desc->y_data_size);
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyGlobalAvgPoolDescriptor(GlobalAvgPoolCudaDescriptor_t desc) {
    if (desc->ndim <= 5) {
        checkCudnnError(cudnnDestroyTensorDescriptor(desc->x_desc));
        checkCudnnError(cudnnDestroyTensorDescriptor(desc->y_desc));
        checkCudnnError(cudnnDestroyPoolingDescriptor(desc->pool_desc));
    }
    desc->cudnn_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
