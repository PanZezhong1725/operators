#include "transpose.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateTransposeDescriptor(CudaHandle_t handle,
                                               TransposeCudaDescriptor_t *desc_ptr,
                                               infiniopTensorDescriptor_t y,
                                               infiniopTensorDescriptor_t x,
                                               uint64_t const *perm,
                                               uint64_t n) {
    uint64_t ndim = y->ndim;
    if (ndim != x->ndim || ndim != n) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t y_data_size = std::accumulate(y->shape, y->shape + y->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t x_data_size = std::accumulate(x->shape, x->shape + x->ndim, 1ULL, std::multiplies<uint64_t>());
    if (y_data_size != x_data_size) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    infiniopTensorDescriptor_t transposed_tensor = permute(x, {perm, perm + n});
    if (!transposed_tensor) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (transposed_tensor->shape[i] != y->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }

    TransposeMode mode = TransposeMode::NON_CONTIGUOUS_COPY;
    bool y_is_contiguous = is_contiguous(y);
    bool transposed = false;

    // determine whether the tranpose operation can be directly collapsed into copying
    if (ndim < 2 || can_squeeze_to_1D(x->shape, ndim) || is_same(perm, n)) {
        mode = (y_is_contiguous && is_contiguous(x)) ? TransposeMode::FULL_CONTIGUOUS_COPY : y_is_contiguous ? TransposeMode::OUTPUT_CONTIGUOUS_COPY
                                                                                                             : TransposeMode::NON_CONTIGUOUS_COPY;
    } else {
        transposed = true;
        mode = y_is_contiguous ? TransposeMode::OUTPUT_CONTIGUOUS_COPY : TransposeMode::NON_CONTIGUOUS_COPY;
    }

    // if perm is not provided, by default, tranpose should reverse the dimensions
    if (!perm && !are_reverse(x->shape, y->shape, ndim)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    size_t shape_size = ndim * sizeof(*x->shape);
    size_t stride_size = ndim * sizeof(*x->strides);
    char *strides_and_shape_d = nullptr;
    checkCudaErrorWithCode(cudaMalloc(&strides_and_shape_d, 2 * (shape_size + stride_size)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMemcpy(strides_and_shape_d, transposed ? transposed_tensor->strides : x->strides, stride_size, cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(strides_and_shape_d + stride_size, y->strides, stride_size, cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(strides_and_shape_d + 2 * stride_size, transposed ? transposed_tensor->shape : x->shape, shape_size, cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(strides_and_shape_d + 2 * stride_size + shape_size, y->shape, shape_size, cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);

    *desc_ptr = new TransposeCudaDescriptor{
        DevNvGpu,
        y->dt,
        handle->device_id,
        ndim,
        y_data_size,
        static_cast<uint64_t>(handle->prop.maxGridSize[0]),
        strides_and_shape_d,
        shape_size,
        stride_size,
        mode,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyTransposeDescriptor(TransposeCudaDescriptor_t desc) {
    checkCudaErrorWithCode(cudaFree((void *) desc->strides_and_shape_d), STATUS_EXECUTION_FAILED);
    delete desc;
    return STATUS_SUCCESS;
}
