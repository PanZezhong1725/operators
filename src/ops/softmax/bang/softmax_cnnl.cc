#include "softmax_cnnl.h"
#include "../../../devices/bang/common_bang.h"
#include "../../../devices/bang/handle_pool.h"
#include "../../utils.h"
#include "cnrt.h"
infiniopStatus_t cnnlCreateSoftmaxDescriptor(BangHandle_t handle,
                                             SoftmaxCnnlDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t input_desc,
                                             infiniopTensorDescriptor_t output_desc) {

    int ndim = input_desc->ndim;
    int *shape = new int[ndim];
    for (int i = 0; i < ndim; i++) {
        shape[i] = static_cast<int>(input_desc->shape[i]);
    }
    *desc_ptr = new SoftmaxCnnlDescriptor{
        handle->device,
        handle->device_id,
        input_desc->dt,
        handle->cnnl_handles,
        ndim,
        shape};
    return STATUS_SUCCESS;
}


infiniopStatus_t cnnlDestroySoftmaxDescriptor(SoftmaxCnnlDescriptor_t desc) {
    desc->cnnl_handles = nullptr;
    //cnnlDestroyTensorDescriptor(desc->aDesc);
    //cnnlDestroyTensorDescriptor(desc->cDesc);
    delete[] desc->shape;
    delete desc;
    return STATUS_SUCCESS;
}

void softmax_cnnl_f16(SoftmaxCnnlDescriptor_t desc, void const *input, int axis, void *output, void *stream) {
    int ndim = desc->ndim;
    auto shape = desc->shape;
    cnnlSoftmaxMode_t mode;
    std::vector<int> inDim = {1, 1, 1};
    std::vector<int> outDim = inDim;

    if (ndim >= 3) {
        if (axis == 0) {
            mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
            inDim[0] = shape[0];
            inDim[1] = shape[1];
            for (int i = 2; i < ndim; ++i) {
                inDim[2] *= shape[i];
            }
            outDim = inDim;
        } else if (axis == ndim - 1) {
            mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
            inDim[0] = shape[0];
            for (int i = 1; i < axis; ++i) {
                inDim[1] *= shape[i];
            }
            inDim[2] = shape[axis];
            outDim = inDim;
        } else {
            mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
            for (int i = 0; i < axis; ++i) {
                inDim[0] *= shape[i];
            }
            inDim[1] = shape[axis];
            for (int i = axis + 1; i < ndim; ++i) {
                inDim[2] *= shape[i];
            }
            outDim = inDim;
        }
    } else if (ndim == 2) {
        if (axis == 0) {
            mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
            inDim[0] = shape[0];
            inDim[1] = shape[1];

            outDim = inDim;
        } else {
            mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
            inDim[1] = shape[0];
            inDim[2] = shape[1];

            outDim = inDim;
        }
    } else {
        mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
        inDim[0] = shape[0];

        outDim = inDim;
    }
    cnnlTensorDescriptor_t aDesc, cDesc;
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlCreateTensorDescriptor(&cDesc);
    cnnlSetTensorDescriptor(
        aDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
        inDim.size(), inDim.data());
    cnnlSetTensorDescriptor(
        cDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
        outDim.size(), outDim.data());

    float alpha = 1.0;
    float beta = 0.0;
    use_cnnl(desc->cnnl_handles, desc->device_id, (cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 cnnlSoftmaxForward_v2(handle, CNNL_SOFTMAX_ACCURATE,
                                       mode, CNNL_COMPUTATION_ULTRAHIGH_PRECISION,
                                       &alpha, aDesc, input, &beta, cDesc, output);
             });
}
infiniopStatus_t cnnlSoftmax(SoftmaxCnnlDescriptor_t desc, void const *input, int axis, void *output, void *stream) {
    if (cnrtSetDevice(desc->device_id) != cnrtSuccess) {
        return STATUS_BAD_DEVICE;
    }

    if (dtype_eq(desc->dtype, F16)) {
        softmax_cnnl_f16(desc, input, axis, output, stream);

        return STATUS_SUCCESS;
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
