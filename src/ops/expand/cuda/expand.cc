#include "expand.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateExpandDescriptor(CudaHandle_t handle,
                                            ExpandCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x) {
    uint64_t ndim = y->ndim;
    if (!isValidBroadcastShape(y, x)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t y_data_size = std::accumulate(y->shape, y->shape + y->ndim, 1ULL, std::multiplies<uint64_t>());

    // get the adjusted strides for x in terms of y
    int64_t *x_strides = new int64_t[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        x_strides[i] = (i < ndim - x->ndim || y->shape[i] != x->shape[i + x->ndim - ndim]) ? 0 : x->strides[i + x->ndim - ndim];
    }

    int64_t *x_strides_d, *y_strides_d;
    char *strides_and_shape_d;
    checkCudaErrorWithCode(cudaMalloc((void **) &strides_and_shape_d, ndim * (2 * sizeof(int64_t) + sizeof(uint64_t))), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMemcpy(strides_and_shape_d, x_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(strides_and_shape_d + ndim * sizeof(int64_t), y->strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(strides_and_shape_d + 2 * ndim * sizeof(int64_t), y->shape, ndim * sizeof(uint64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);

    *desc_ptr = new ExpandCudaDescriptor{
        DevNvGpu,
        y->dt,
        handle->device_id,
        ndim,
        y_data_size,
        static_cast<uint64_t>(handle->prop.maxGridSize[0]),
        strides_and_shape_d,
    };

    delete[] x_strides;

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyExpandDescriptor(ExpandCudaDescriptor_t desc) {
    checkCudaErrorWithCode(cudaFree((void *) desc->strides_and_shape_d), STATUS_EXECUTION_FAILED);
    delete desc;
    return STATUS_SUCCESS;
}
