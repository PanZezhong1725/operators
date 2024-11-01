#include "softmax_cuda.cuh"
#include "../../utils.h"

infiniopStatus_t cudaCreateSoftmaxDescriptor(CudaHandle_t handle,
                                             SoftmaxCudaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t output_desc) {

    ASSERT_EQ(input_desc->ndim, output_desc->ndim);
    if (!dtype_eq(input_desc->dt, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    int ndim = input_desc->ndim;

    for (int i = 0; i < ndim; i++) {
        if (input_desc->shape[i] != output_desc->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    int *shape = new int[ndim];


    for (int i = 0; i < ndim; i++) {
        shape[i] = static_cast<int>(input_desc->shape[i]);
    }
    *desc_ptr = new SoftmaxCudaDescriptor{
        handle->device,
        handle->device_id,
        input_desc->dt,
        ndim,
        shape};

    return STATUS_SUCCESS;
}


infiniopStatus_t cudaDestroySoftmaxDescriptor(SoftmaxCudaDescriptor_t desc) {
    delete[] desc->shape;
    delete desc;
    return STATUS_SUCCESS;
}
