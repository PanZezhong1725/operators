#include "rms_norm_aclnn.h"

RMSNormAclnnDescriptor::RMSNormAclnnDescriptor(Device _device) {
    device = _device;
    handle = nullptr;
    workspaceSize = 0;
    yDesc = new aclnnTensorDescriptor();
    xDesc = new aclnnTensorDescriptor();
    wDesc = new aclnnTensorDescriptor();
    rstdDesc = new aclnnTensorDescriptor();
    castDesc = nullptr;
    epsilon = 1e-5;
}


infiniopStatus_t aclnnCreateRMSNormDescriptor(AscendHandle_t handle,
                                              RMSNormAclnnDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t y,
                                              infiniopTensorDescriptor_t x,
                                              infiniopTensorDescriptor_t w,
                                              float eps) {
    *desc_ptr = new RMSNormAclnnDescriptor(handle->device);
    (*desc_ptr)->handle = reinterpret_cast<AscendHandle_t>(handle);
    (*desc_ptr)->epsilon = static_cast<double>(eps);

    auto &yDesc = (*desc_ptr)->yDesc;
    auto &xDesc = (*desc_ptr)->xDesc;
    auto &wDesc = (*desc_ptr)->wDesc;
    auto &castDesc = (*desc_ptr)->castDesc;

    auto status = yDesc->fromInfiniOpTensorDescriptor(y);
    status = xDesc->fromInfiniOpTensorDescriptor(x);
    status = wDesc->fromInfiniOpTensorDescriptor(w);

    // Set rstdDesc
    // See: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/apiref/appdevgapi/context/aclnnRmsNorm.md
    // rstdTensor cannot set nullptr in aclnn
    int64_t wsize = 1;
    for (uint64_t i = 0; i < wDesc->ndim; ++i) {
        wsize *= (wDesc->shape)[i];
    }
    int64_t xsize = 1;
    uint64_t rstd_dim = xDesc->ndim - 1;
    for (int64_t i = xDesc->ndim - 1; i >= 0; --i) {
        xsize *= (xDesc->shape)[i];
        rstd_dim = static_cast<uint64_t>(i);
        if (xsize == wsize) {
            break;
        }
    }

    auto rstd_shape = new std::vector<int64_t>(xDesc->ndim, 1);
    auto rstd_strides = new std::vector<int64_t>(xDesc->ndim, 1);

    for (uint64_t i = 0; i < rstd_dim; ++i) {
        (*rstd_shape)[i] = (xDesc->shape)[i];
    }
    for (int64_t i = xDesc->ndim - 2; i >= 0; --i) {
        (*rstd_strides)[i] = (*rstd_strides)[i + 1] * (*rstd_shape)[i + 1];
    }

    auto &rstdDesc = (*desc_ptr)->rstdDesc;
    rstdDesc->ndim = rstd_shape->size();
    rstdDesc->shape = rstd_shape->data();
    rstdDesc->strides = rstd_strides->data();
    rstdDesc->offset = 0;
    // Only support FLOAT
    rstdDesc->dataType = aclDataType::ACL_FLOAT;
    rstdDesc->storageShape = rstd_shape->data();
    rstdDesc->storageNdim = rstd_shape->size();

    if (wDesc->dataType != xDesc->dataType) {
        castDesc = new aclnnTensorDescriptor();
        status = castDesc->fromInfiniOpTensorDescriptor(w);
        castDesc->dataType = xDesc->dataType;
        status = castDesc->createTensor();
    }

    status = yDesc->createTensor();
    status = xDesc->createTensor();
    status = wDesc->createTensor();
    status = rstdDesc->createTensor();

    return status;
}

infiniopStatus_t aclnnGetRMSNormWorkspaceSize(RMSNormAclnnDescriptor_t desc,
                                              uint64_t *size) {
    auto &yDesc = desc->yDesc;
    auto &xDesc = desc->xDesc;
    auto &wDesc = desc->wDesc;
    auto &rstdDesc = desc->rstdDesc;
    auto &castDesc = desc->castDesc;

    // Get Tensor
    aclTensor *ty = yDesc->t;
    aclTensor *tx = xDesc->t;
    aclTensor *tw = wDesc->t;
    aclTensor *trstd = rstdDesc->t;

    uint64_t workspaceSize;
    auto &handle = desc->handle;

    use_aclnn((AscendHandle_t) handle,
              [&](aclOpExecutor *&executor) {
                  auto ret =
                      aclnnRmsNormGetWorkspaceSize(tx,
                                                   castDesc == nullptr ? tw
                                                                       : castDesc->t,
                                                   desc->epsilon,
                                                   ty,
                                                   trstd,
                                                   &workspaceSize,
                                                   &executor);
                  aclSetAclOpExecutorRepeatable(executor);
                  CHECK_RET(ret == ACL_SUCCESS,
                            LOG_PRINT("aclnnRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret));
              });
    *size = workspaceSize +
            numElements(rstdDesc->shape, rstdDesc->ndim) * aclDataTypeSize(rstdDesc->dataType);

    if (castDesc != nullptr) {
        *size += numElements(castDesc->shape, castDesc->ndim) * aclDataTypeSize(castDesc->dataType);
    }

    desc->workspaceSize = workspaceSize;

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnRMSNorm(RMSNormAclnnDescriptor_t desc,
                              void *workspace,
                              uint64_t workspace_size,
                              void *y,
                              void *x,
                              void *w,
                              void *stream) {
    auto &yDesc = desc->yDesc;
    auto &xDesc = desc->xDesc;
    auto &wDesc = desc->wDesc;
    auto &rstdDesc = desc->rstdDesc;
    auto &castDesc = desc->castDesc;

    // Get Tensor
    aclTensor *ty = yDesc->t;
    aclTensor *tx = xDesc->t;
    aclTensor *tw = wDesc->t;
    aclTensor *trstd = rstdDesc->t;

    auto rstd = (void *) ((uint8_t *) workspace + desc->workspaceSize);
    auto &handle = desc->handle;
    void *castPtr = nullptr;

    if (castDesc != nullptr) {
        aclTensor *tcast = castDesc->t;
        castPtr = (void *) ((float *) rstd + numElements(rstdDesc->shape, rstdDesc->ndim));

        aclOpExecutor *castExecutor = nullptr;
        uint64_t workspaceSize = 0;
        auto ret = aclnnCastGetWorkspaceSize(tw, castDesc->dataType, tcast, &workspaceSize, &castExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret));
        aclSetAclOpExecutorRepeatable(castExecutor);

        AclSetTensorAddr(castExecutor, 0, tw, w);
        AclSetTensorAddr(castExecutor, 1, tcast, castPtr);
        ret = aclnnCast(nullptr, workspaceSize, castExecutor, stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret));
        aclDestroyAclOpExecutor(castExecutor);
    }

    use_aclnn((AscendHandle_t) handle,
              [&](aclOpExecutor *executor) {
                  AclSetTensorAddr(executor, 0, tx, x);
                  if (castDesc != nullptr) {
                      AclSetTensorAddr(executor, 1, castDesc->t, castPtr);
                  } else {
                      AclSetTensorAddr(executor, 1, tw, w);
                  }
                  AclSetTensorAddr(executor, 2, ty, y);
                  AclSetTensorAddr(executor, 3, trstd, rstd);

                  auto ret = aclnnRmsNorm(workspace,
                                          desc->workspaceSize,
                                          executor,
                                          stream);
                  aclDestroyAclOpExecutor(executor);
                  CHECK_RET(ret == ACL_SUCCESS,
                            LOG_PRINT("aclnnRmsNorm failed. ERROR: %d\n", ret));
              });

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnDestroyRMSNormDescriptor(RMSNormAclnnDescriptor_t desc) {
    delete desc->yDesc;
    delete desc->wDesc;
    delete desc->xDesc;
    delete desc->rstdDesc;
    if (desc->castDesc) {
        delete desc->castDesc;
    }

    return STATUS_SUCCESS;
}
