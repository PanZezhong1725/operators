#include "causal_softmax_cnnl.h"
#include "../../../devices/bang/bang_handle.h"
#include "../../../devices/bang/common_bang.h"
#include "../../../utils.h"
#include "cnnl_extra.h"

infiniopStatus_t cnnlCreateCausalSoftmaxDescriptor(BangHandle_t handle,
                                                   CausalSoftmaxCnnlDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y) {
    if (y->ndim < 2 || y->shape[y->ndim - 1] < y->shape[y->ndim - 2]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    // cnnlMaskedSoftmax only support 4D or 5D tensors
    int ndim_ = std::max(static_cast<int>(y->ndim), 4);
    std::vector<int> dims(ndim_, 1);
    for (uint64_t i = 0; i < y->ndim; i++) {
        dims[ndim_ - 1 - i] = static_cast<int>(y->shape[y->ndim - i - 1]);
    }

    cnnlTensorDescriptor_t yDesc, maskDesc;
    cnnlCreateTensorDescriptor(&yDesc);
    cnnlCreateTensorDescriptor(&maskDesc);
    cnnlSetTensorDescriptor(yDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(y->dt),
                            dims.size(), dims.data());
    cnnlSetTensorDescriptor(maskDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL,
                            dims.size(), dims.data());

    *desc_ptr = new CausalSoftmaxCnnlDescriptor{
        handle->device,
        handle->device_id,
        handle->cnnl_handles,
        y->dt,
        std::move(yDesc),
        std::move(maskDesc),
        std::move(dims)};

    return STATUS_SUCCESS;
}

infiniopStatus_t cnnlGetCausalSoftmaxWorkspaceSize(CausalSoftmaxCnnlDescriptor_t desc, unsigned long int *size) {
    *size = sizeof(bool) * desc->dims[0] * desc->dims[1] * desc->dims[2] * desc->dims[3];
    return STATUS_SUCCESS;
}

infiniopStatus_t cnnlDestroyCausalSoftmaxDescriptor(CausalSoftmaxCnnlDescriptor_t desc) {
    cnnlDestroyTensorDescriptor(desc->yDesc);
    cnnlDestroyTensorDescriptor(desc->maskDesc);
    delete desc;
    return STATUS_SUCCESS;
}

infiniopStatus_t cnnlCausalSoftmax(CausalSoftmaxCnnlDescriptor_t desc,
                                   void *workspace,
                                   unsigned long int workspace_size,
                                   void *data,
                                   void *stream) {
    if (cnrtSetDevice(desc->device_id) != cnrtSuccess) {
        return STATUS_BAD_DEVICE;
    }
    bool mask_matrix[desc->dims[0]][desc->dims[1]][desc->dims[2]][desc->dims[3]];

    // 填充上三角矩阵（右上角为 false）
    for (int i = 0; i < desc->dims[0]; ++i) {
        for (int j = 0; j < desc->dims[1]; ++j) {
            for (int m = 0; m < desc->dims[2]; ++m) {
                for (int n = 0; n < desc->dims[3]; ++n) {
                    if (n - m > desc->dims[3] - desc->dims[2]) {
                        mask_matrix[i][j][m][n] = true;
                    } else {
                        mask_matrix[i][j][m][n] = false;
                    }
                }
            }
        }
    }

    cnrtMemcpyAsync(workspace, mask_matrix, workspace_size, (cnrtQueue_t) stream, cnrtMemcpyHostToDev);

    use_cnnl(desc->pool, desc->device_id, (cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 cnnlMaskedSoftmax(handle, CNNL_MASKED_SOFTMAX_MASKED_FILL,
                                   -1, 1.0, desc->yDesc, data, desc->maskDesc, workspace,
                                   desc->yDesc, data);
             });

    return STATUS_SUCCESS;
}
