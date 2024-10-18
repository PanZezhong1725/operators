#include "conv.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateConvDescriptor(CudaHandle_t handle,
                                          ConvCudaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x,
                                          infiniopTensorDescriptor_t w,
                                          void const *pads,
                                          void const *strides,
                                          void const *dilations,
                                          uint64_t n,
                                          int device_id) {
    uint64_t ndim = y->ndim;
    if (ndim < 3 || ndim != x->ndim || ndim != w->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (x->shape[0] != y->shape[0] || w->shape[0] != y->shape[1] || x->shape[1] != w->shape[1]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt || y->dt != w->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    // convert pads, strides, dilations into int32[]
    int32_t *pad = new int32_t[ndim];
    int32_t *stride = new int32_t[ndim];
    int32_t *dilation = new int32_t[ndim];
    int32_t *x_shape = new int32_t[ndim];
    int32_t *w_shape = new int32_t[ndim];
    int32_t *y_shape = new int32_t[ndim];
    auto pads_ = reinterpret_cast<uint64_t const *>(pads);
    auto strides_ = reinterpret_cast<int64_t const *>(strides);
    auto dilations_ = reinterpret_cast<uint64_t const *>(dilations);
    for (size_t i = 0; i < ndim; ++i) {
        pad[i] = static_cast<int32_t>(pads_[i]);
        stride[i] = static_cast<int32_t>(strides_[i]);
        dilation[i] = static_cast<int32_t>(dilations_[i]);
        x_shape[i] = static_cast<int32_t>(x->shape[i]);
        w_shape[i] = static_cast<int32_t>(w->shape[i]);
        y_shape[i] = static_cast<int32_t>(y->shape[i]);
    }

    // get the data types of the tensors and the conv operator
    CREATE_CHECK_ERROR(auto tensor_dt = dataTypeMap[x->dt], tensor_dt, -1, STATUS_BAD_PARAM);
    cudnnDataType_t conv_op_dt = [&] {
        switch (tensor_dt) {
            case CUDNN_DATA_HALF:
            case CUDNN_DATA_BFLOAT16:
            case CUDNN_DATA_FLOAT:
                return CUDNN_DATA_FLOAT;
            case CUDNN_DATA_DOUBLE:
                return CUDNN_DATA_DOUBLE;
            default:
                return CUDNN_DATA_INT32;
        }
    }();

    // create and set tensor descriptors for x
    cudnnTensorDescriptor_t x_desc;
    checkCudnnError(cudnnCreateTensorDescriptor(&x_desc));
    checkCudnnError(cudnnSetTensorNdDescriptorEx(x_desc, CUDNN_TENSOR_NCHW, static_cast<cudnnDataType_t>(tensor_dt), ndim, x_shape));

    // create and set tensor descriptors for w
    cudnnFilterDescriptor_t w_desc;
    checkCudnnError(cudnnCreateFilterDescriptor(&w_desc));
    checkCudnnError(cudnnSetFilterNdDescriptor(w_desc, static_cast<cudnnDataType_t>(tensor_dt), CUDNN_TENSOR_NCHW, ndim, w_shape));

    // create and set conv operator descriptor
    cudnnConvolutionDescriptor_t op_desc;
    checkCudnnError(cudnnCreateConvolutionDescriptor(&op_desc));
    checkCudnnError(cudnnSetConvolutionNdDescriptor(
        op_desc, ndim - 2, pad, stride, dilation, CUDNN_CROSS_CORRELATION,
        conv_op_dt));

    // create and set tensor descriptors for y
    cudnnTensorDescriptor_t y_desc;
    int outDim[ndim];
    checkCudnnError(cudnnGetConvolutionNdForwardOutputDim(op_desc, x_desc, w_desc, ndim, outDim));
    checkCudnnError(cudnnCreateTensorDescriptor(&y_desc));
    checkCudnnError(cudnnSetTensorNdDescriptorEx(y_desc, CUDNN_TENSOR_NCHW, static_cast<cudnnDataType_t>(tensor_dt), ndim, y_shape));

    // get the best algorithm
    const int requestedAlgoCount = 1;
    int algoCounts;
    cudnnConvolutionFwdAlgoPerf_t perf_results[requestedAlgoCount];
    checkCudnnError(use_cudnn(handle->cudnn_handles_t, device_id,
                              [&](cudnnHandle_t handle) { return cudnnFindConvolutionForwardAlgorithm(handle, x_desc, w_desc, op_desc, y_desc, requestedAlgoCount, &algoCounts, perf_results); }));
    if (algoCounts < 1) {
        return STATUS_EXECUTION_FAILED;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    *desc_ptr = new ConvCudaDescriptor{
        DevNvGpu,
        y->dt,
        device_id,
        handle->cudnn_handles_t,
        x_desc,
        w_desc,
        y_desc,
        op_desc,
        perf_results[0].algo,
        alpha,
        beta};

    delete[] pad;
    delete[] stride;
    delete[] dilation;
    delete[] x_shape;
    delete[] w_shape;
    delete[] y_shape;

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGetConvWorkspaceSize(ConvCudaDescriptor_t desc, uint64_t *size) {
    checkCudnnError(use_cudnn(desc->cudnn_handles_t, desc->device_id,
                              [&](cudnnHandle_t handle) { return cudnnGetConvolutionForwardWorkspaceSize(handle, desc->x_desc, desc->w_desc, desc->op_desc, desc->y_desc, desc->algo, size); }));
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyConvDescriptor(ConvCudaDescriptor_t desc) {
    checkCudnnError(cudnnDestroyConvolutionDescriptor(desc->op_desc));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->y_desc));
    checkCudnnError(cudnnDestroyFilterDescriptor(desc->w_desc));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->x_desc));
    desc->cudnn_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
