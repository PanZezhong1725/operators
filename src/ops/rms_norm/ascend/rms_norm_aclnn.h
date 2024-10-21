#ifndef __ACLNN_RMS_NORM_H__
#define __ACLNN_RMS_NORM_H__

#include "../../../devices/ascend/ascend_handle.h"
#include "../../../devices/ascend/tensor_aclnn.h"
#include "../../utils.h"
#include "operators.h"
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_cast.h>
#include <algorithm>

struct RMSNormAclnnDescriptor {
    Device device;
    AscendHandle_t handle;
    aclOpExecutor *executor;
    aclnnTensorDescriptor_t yDesc, xDesc, wDesc, rstdDesc, castDesc;
    uint64_t workspaceSize;
    double epsilon;

    RMSNormAclnnDescriptor(Device device);
};

typedef RMSNormAclnnDescriptor *RMSNormAclnnDescriptor_t;

infiniopStatus_t aclnnCreateRMSNormDescriptor(AscendHandle_t handle,
                                              RMSNormAclnnDescriptor_t *desc,
                                              infiniopTensorDescriptor_t y,
                                              infiniopTensorDescriptor_t x,
                                              infiniopTensorDescriptor_t w,
                                              float eps);

infiniopStatus_t aclnnGetRMSNormWorkspaceSize(RMSNormAclnnDescriptor_t desc,
                                              uint64_t *size);

infiniopStatus_t aclnnRMSNorm(RMSNormAclnnDescriptor_t desc,
                              void *workspace,
                              uint64_t workspace_size,
                              void *y,
                              void *x,
                              void *w,
                              void *stream);

infiniopStatus_t aclnnDestroyRMSNormDescriptor(RMSNormAclnnDescriptor_t desc);

#endif