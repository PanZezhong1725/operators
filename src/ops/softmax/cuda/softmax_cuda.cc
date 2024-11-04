#include "../../utils.h"
#include "softmax.cuh"

infiniopStatus_t cudaCreateSoftmaxDescriptor(CudaHandle_t handle,
                                             SoftmaxCudaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t input_desc, int axis, infiniopTensorDescriptor_t output_desc) {

    ASSERT_EQ(input_desc->ndim, output_desc->ndim);
    if (!dtype_eq(input_desc->dt, F16) && !dtype_eq(input_desc->dt, F32)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    int ndim = input_desc->ndim;

    for (int i = 0; i < ndim; i++) {
        if (input_desc->shape[i] != output_desc->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    int dimsize = 1;
    int stride = 1;
    int othersize = 1;
    int size = 1;
    for (int i = ndim - 1; i >= 0; i -= 1) {
        size *= static_cast<int>(input_desc->shape[i]);
    }
    for (int i = ndim - 1; i >= 0; i -= 1) {
        if (i == axis) {
            break;
        }
        stride *= static_cast<int>(input_desc->shape[i]);
    }
    *desc_ptr = new SoftmaxCudaDescriptor{
        handle->device,
        handle->device_id,
        input_desc->dt,
        dimsize,
        stride,
        othersize};

    return STATUS_SUCCESS;
}


infiniopStatus_t cudaDestroySoftmaxDescriptor(SoftmaxCudaDescriptor_t desc) {

    delete desc;
    return STATUS_SUCCESS;
}
