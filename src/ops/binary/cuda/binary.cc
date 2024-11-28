#include "binary.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateBinaryDescriptor(CudaHandle_t handle,
                                            BinaryCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c,
                                            infiniopTensorDescriptor_t a,
                                            infiniopTensorDescriptor_t b,
                                            int mode) {
    uint64_t ndim = c->ndim;
    if (!isValidBroadcastShape(a, b, c)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(c)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (c->dt != F16 && c->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (c->dt != a->dt || c->dt != b->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    bool broadcasted = false;
    if (ndim != a->ndim || ndim != b->ndim) {
        broadcasted = true;
    } else {
        for (uint64_t i = 0; i < ndim; ++i) {
            if (c->shape[i] != a->shape[i] || c->shape[i] != b->shape[i]) {
                broadcasted = true;
                break;
            }
        }
    }
    if (mode < 0 || mode >= BinaryMode::numBinaryMode) {
        return STATUS_BAD_PARAM;
    }
    // bitwise operations are only valid for integral types
    if (c->dt.exponent != 0 && (mode == BinaryMode::BitwiseAnd || mode == BinaryMode::BitwiseOr || mode == BinaryMode::BitwiseXor)) {
        return STATUS_BAD_PARAM;
    }

    uint64_t c_data_size = std::accumulate(c->shape, c->shape + c->ndim, 1ULL, std::multiplies<uint64_t>());

    char *strides_d = nullptr;
    size_t stride_arr_size = ndim * sizeof(int64_t);
    if (broadcasted) {
        // get the adjusted strides for a and b
        int64_t a_strides[ndim];
        int64_t b_strides[ndim];
        for (size_t i = 0; i < ndim; ++i) {
            a_strides[i] = (i < ndim - a->ndim || c->shape[i] != a->shape[i + a->ndim - ndim]) ? 0 : a->strides[i + a->ndim - ndim];
            b_strides[i] = (i < ndim - b->ndim || c->shape[i] != b->shape[i + b->ndim - ndim]) ? 0 : b->strides[i + b->ndim - ndim];
        }

        // malloc and copy the strides to the device
        checkCudaErrorWithCode(cudaMalloc(&strides_d, 3 * stride_arr_size), STATUS_MEMORY_NOT_ALLOCATED);
        checkCudaErrorWithCode(cudaMemcpy(strides_d, a_strides, stride_arr_size, cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
        checkCudaErrorWithCode(cudaMemcpy(strides_d + stride_arr_size, b_strides, stride_arr_size, cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
        checkCudaErrorWithCode(cudaMemcpy(strides_d + 2 * stride_arr_size, c->strides, stride_arr_size, cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    }

    *desc_ptr = new BinaryCudaDescriptor{
        DevNvGpu,
        c->dt,
        handle->device_id,
        ndim,
        c_data_size,
        static_cast<uint64_t>(handle->prop.maxGridSize[0]),
        reinterpret_cast<int64_t const *>(strides_d),
        broadcasted,
        mode,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyBinaryDescriptor(BinaryCudaDescriptor_t desc) {
    checkCudaErrorWithCode(cudaFree((void *) desc->strides_d), STATUS_EXECUTION_FAILED);
    delete desc;
    return STATUS_SUCCESS;
}
