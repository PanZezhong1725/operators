#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "random_sample.cuh"

infiniopStatus_t cudaCreateRandomSampleDescriptor(CudaHandle_t handle,
                                                  RandomSampleCudaDescriptor_t *desc_ptr,
                                                  infiniopTensorDescriptor_t probs) {
    if (probs->ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    int voc = probs->shape[0];

    *desc_ptr = new RandomSampleCudaDescriptor{
        handle->device,
        handle->device_id,
        probs->dt,
        voc};

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGetRandomSampleWorkspaceSize(RandomSampleCudaDescriptor_t desc, unsigned long int *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyRandomSampleDescriptor(RandomSampleCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
