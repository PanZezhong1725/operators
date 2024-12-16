#include "causal_softmax_bang.h"
#include "../../utils.h"

infiniopStatus_t bangCreateCausalSoftmaxDescriptor(BangHandle_t handle,
                                                   CausalSoftmaxBangDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y) {
    if (y->ndim < 2 || y->shape[y->ndim - 1] < y->shape[y->ndim - 2]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    int ndim = y->ndim;
    int *stride = new int[ndim];
    int *shape = new int[ndim];

    int n = 1;
    for (int i = 0; i < ndim; i++) {
        stride[i] = static_cast<int>(y->strides[i]);
        shape[i] = static_cast<int>(y->shape[i]);
        if (i < ndim - 1) {
            n *= shape[i];
        }
    }

    *desc_ptr = new CausalSoftmaxBangDescriptor{
        handle->device,
        handle->device_id,
        y->dt,
        ndim,
        stride,
        shape,
        n};

    return STATUS_SUCCESS;
}

infiniopStatus_t bangGetCausalSoftmaxWorkspaceSize(CausalSoftmaxBangDescriptor_t desc, uint64_t *size) {
    if (desc->ndim > 3) {
        *size = desc->ndim * sizeof(int) * 2;
    } else {
        *size = 0;
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t bangDestroyCausalSoftmaxDescriptor(CausalSoftmaxBangDescriptor_t desc) {
    delete[] desc->stride;
    delete[] desc->shape;
    delete desc;
    return STATUS_SUCCESS;
}
