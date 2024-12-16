#include "add.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateAddDescriptor(CudaHandle_t handle,
                                         AddCudaDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t c,
                                         infiniopTensorDescriptor_t a,
                                         infiniopTensorDescriptor_t b) {
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

    uint64_t c_data_size = std::accumulate(c->shape, c->shape + c->ndim, 1ULL, std::multiplies<uint64_t>());

    // get the adjusted strides for a and b
    int64_t *a_strides = new int64_t[ndim];
    int64_t *b_strides = new int64_t[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        a_strides[i] = (i < ndim - a->ndim || c->shape[i] != a->shape[i + a->ndim - ndim]) ? 0 : a->strides[i + a->ndim - ndim];
        b_strides[i] = (i < ndim - b->ndim || c->shape[i] != b->shape[i + b->ndim - ndim]) ? 0 : b->strides[i + b->ndim - ndim];
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, handle->device_id);

    int64_t *a_strides_d, *b_strides_d, *c_strides_d;
    checkCudaErrorWithCode(cudaMalloc((void **) &a_strides_d, ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void **) &b_strides_d, ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void **) &c_strides_d, ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMemcpy(a_strides_d, a_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(b_strides_d, b_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(c_strides_d, c->strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);

    *desc_ptr = new AddCudaDescriptor{
        DevNvGpu,
        c->dt,
        handle->device_id,
        ndim,
        c_data_size,
        static_cast<uint64_t>(prop.maxGridSize[0]),
        a_strides_d,
        b_strides_d,
        c_strides_d,
        broadcasted,
    };

    delete[] a_strides;
    delete[] b_strides;

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyAddDescriptor(AddCudaDescriptor_t desc) {
    checkCudaErrorWithCode(cudaFree((void *) desc->a_strides), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void *) desc->b_strides), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void *) desc->c_strides), STATUS_EXECUTION_FAILED);
    delete desc;
    return STATUS_SUCCESS;
}
