#include "causal_softmax_aclnn.h"
#include "../../../utils.h"

CausalSoftmaxAclnnDescriptor::CausalSoftmaxAclnnDescriptor(Device _device) {
    device = _device;
    handle = nullptr;
    aDesc = new aclnnTensorDescriptor();
    maskDesc = new aclnnTensorDescriptor();
    outDesc = new aclnnTensorDescriptor();
    workspaceSize = 0;
}

infiniopStatus_t aclnnCreateCausalSoftmaxDescriptor(AscendHandle_t handle,
                                                    CausalSoftmaxAclnnDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t y) {
    // Construct CausalSoftmaxAclnnDescriptor
    *desc_ptr = new CausalSoftmaxAclnnDescriptor(handle->device);
    (*desc_ptr)->handle = reinterpret_cast<AscendHandle_t>(handle);

    // Set value from infiniopTensorDescriptor
    auto &aDesc = (*desc_ptr)->aDesc;
    auto &outDesc = (*desc_ptr)->outDesc;

    uint64_t ndim = y->ndim;
    uint64_t *shape = y->shape;
    int64_t *strides = y->strides;
    int64_t total_seq_len = static_cast<int64_t>(shape[ndim - 1]);
    int64_t seq_len = static_cast<int64_t>(shape[ndim - 2]);

    if (total_seq_len < seq_len) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    // Change input shape and stride
    auto aclnn_shape = new std::vector<uint64_t>(4);
    auto aclnn_strides = new std::vector<int64_t>(4);
    for (uint64_t i = 0; i < ndim; ++i) {
        (*aclnn_shape)[4 - i - 1] = shape[ndim - i - 1];
        (*aclnn_strides)[4 - i - 1] = strides[ndim - i - 1];
    }
    for (uint64_t i = 0; i < 4 - ndim; ++i) {
        (*aclnn_shape)[i] = 1;
        (*aclnn_strides)[i] = (*aclnn_shape)[i + 1] * (*aclnn_strides)[i + 1];
    }

    auto _y = y;
    _y->shape = aclnn_shape->data();
    _y->ndim = aclnn_shape->size();
    _y->strides = aclnn_strides->data();

    auto status = aDesc->fromInfiniOpTensorDescriptor(_y);
    status = outDesc->fromInfiniOpTensorDescriptor(_y);

    // Set mask Desc
    auto &maskDesc = (*desc_ptr)->maskDesc;
    auto mask_shape = new std::vector<int64_t>(3);

    (*mask_shape)[2] = total_seq_len;
    (*mask_shape)[1] = seq_len;
    if (ndim == 2) {
        (*mask_shape)[0] = 1;
    } else {
        (*mask_shape)[0] = static_cast<int64_t>(shape[0]);
    }
    auto mask_strides = new std::vector<int64_t>{total_seq_len * seq_len, total_seq_len, 1};


    maskDesc->ndim = mask_shape->size();
    maskDesc->shape = mask_shape->data();
    maskDesc->strides = mask_strides->data();
    maskDesc->offset = 0;
    maskDesc->dataType = aDesc->dataType;
    maskDesc->format = aDesc->format;
    maskDesc->storageShape = mask_shape->data();
    maskDesc->storageNdim = mask_shape->size();

    // Create aclTensor
    status = aDesc->createTensor();
    status = maskDesc->createTensor();
    status = outDesc->createTensor();

    return status;
}

infiniopStatus_t aclnnGetCausalSoftmaxWorkspaceSize(CausalSoftmaxAclnnDescriptor_t desc, uint64_t *size) {
    auto &maskDesc = desc->maskDesc;
    auto &aDesc = desc->aDesc;
    auto &outDesc = desc->outDesc;
    auto &handle = desc->handle;

    // Get Tensor
    aclTensor *ta = aDesc->t;
    aclTensor *tmask = maskDesc->t;
    aclTensor *tout = outDesc->t;

    uint64_t workspaceSize;

    use_aclnn((AscendHandle_t) handle,
              [&](aclOpExecutor *&executor) {
                  auto ret =
                      aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(ta,
                                                                       nullptr,
                                                                       tmask,
                                                                       1.0, 0,
                                                                       tout,
                                                                       &workspaceSize,
                                                                       &executor);
                  aclSetAclOpExecutorRepeatable(executor);
                  CHECK_RET(ret == ACL_SUCCESS,
                            LOG_PRINT("aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize failed. ERROR: %d\n", ret));
              });
    *size = workspaceSize +
            numElements(maskDesc->shape, maskDesc->ndim) * aclDataTypeSize(maskDesc->dataType);

    desc->workspaceSize = workspaceSize;

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnCausalSoftmax(CausalSoftmaxAclnnDescriptor_t desc,
                                    void *workspace,
                                    uint64_t workspace_size,
                                    void *data,
                                    void *stream) {
    auto &aDesc = desc->aDesc;
    auto &maskDesc = desc->maskDesc;
    auto &outDesc = desc->outDesc;
    auto &handle = desc->handle;

    // Get aclTensor pt
    aclTensor *ta = aDesc->t;
    aclTensor *tmask = maskDesc->t;
    aclTensor *tout = outDesc->t;

    // Fill upgrade matrix
    uint16_t mask_matrix[maskDesc->shape[0]][maskDesc->shape[1]][maskDesc->shape[2]];
    auto &dims = maskDesc->shape;
    auto ele_size = aclDataTypeSize(maskDesc->dataType);

    // float neg_inf = -100000000;
    for (int i = 0; i < dims[0]; ++i) {
        for (int m = 0; m < dims[1]; ++m) {
            for (int n = 0; n < dims[2]; ++n) {
                if (n - m > dims[2] - dims[1]) {
                    // 0xF939 = -10240 half
                    mask_matrix[i][m][n] = 0xF880;
                } else {
                    mask_matrix[i][m][n] = 0;
                }
            }
        }
    }

    aclrtMemcpy(workspace,
                workspace_size * ele_size,
                mask_matrix,
                numElements(maskDesc->shape, maskDesc->ndim) * ele_size,
                ACL_MEMCPY_HOST_TO_DEVICE);

    use_aclnn((AscendHandle_t) handle,
              [&](aclOpExecutor *executor) {
                  AclSetTensorAddr(executor, 0, ta, data);
                  AclSetTensorAddr(executor, 2, tmask, workspace);
                  AclSetTensorAddr(executor, 3, tout, data);

                  workspace = (void *) ((uint16_t *) workspace + numElements(maskDesc->shape, maskDesc->ndim));
                  auto ret = aclnnMaskedSoftmaxWithRelPosBias(workspace,
                                                              desc->workspaceSize,
                                                              executor,
                                                              stream);
                  aclDestroyAclOpExecutor(executor);
                  CHECK_RET(ret == ACL_SUCCESS,
                            LOG_PRINT("aclnnMaskedSoftmaxWithRelPosBias failed. ERROR: %d\n", ret));
              });

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnDestroyCausalSoftmaxDescriptor(CausalSoftmaxAclnnDescriptor_t desc) {
    delete desc->aDesc;
    delete desc->maskDesc;
    delete desc->outDesc;
    delete desc;
    return STATUS_SUCCESS;
}
