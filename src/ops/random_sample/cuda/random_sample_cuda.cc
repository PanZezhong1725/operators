#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "random_sample.cuh"

infiniopStatus_t cudaCreateRandomSampleDescriptor(CudaHandle_t handle,
                                                  RandomSampleCudaDescriptor_t *desc_ptr, infiniopTensorDescriptor_t result,
                                                  infiniopTensorDescriptor_t probs) {
    if (probs->ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!dtype_eq(result->dt, U64))
        return STATUS_BAD_TENSOR_DTYPE;
    int voc = probs->shape[0];
    int rLength = result->shape[0];
    if (result->ndim != 1 && rLength != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    *desc_ptr = new RandomSampleCudaDescriptor{
        handle->device,
        handle->device_id,
        probs->dt,
        voc,
        result->dt,
        rLength};

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGetRandomSampleWorkspaceSize(RandomSampleCudaDescriptor_t desc, unsigned long int *size) {
    size_t size_radix_sort;
    size_t size_scan;
    random_sample_workspace(size_radix_sort, size_scan,
                            desc->voc, desc->dtype);
    *size = desc->voc * (2 * sizeof(uint64_t) + sizeof(desc->dtype)) + std::max(size_radix_sort, size_scan);
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyRandomSampleDescriptor(RandomSampleCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
