#include "rearrange_aclnn.h"
#include "../../../utils.h"

RearrangeAclnnDescriptor::RearrangeAclnnDescriptor(Device _device) {
    device = _device;
    handle = nullptr;
    dstDesc = new aclnnTensorDescriptor();
    srcDesc = new aclnnTensorDescriptor();
    workspaceSize = 0;
}

infiniopStatus_t aclnnCreateRearrangeDescriptor(AscendHandle_t handle,
                                                RearrangeAclnnDescriptor_t *desc_ptr,
                                                infiniopTensorDescriptor_t dst,
                                                infiniopTensorDescriptor_t src) {
    *desc_ptr = new RearrangeAclnnDescriptor(handle->device);
    (*desc_ptr)->handle = reinterpret_cast<AscendHandle_t>(handle);

    auto &dstDesc = (*desc_ptr)->dstDesc;
    auto &srcDesc = (*desc_ptr)->srcDesc;

    auto status = dstDesc->fromInfiniOpTensorDescriptor(dst);
    status = srcDesc->fromInfiniOpTensorDescriptor(src);

    status = dstDesc->createTensor();
    status = srcDesc->createTensor();

    return status;
}

infiniopStatus_t aclnnRearrange(RearrangeAclnnDescriptor_t desc,
                                void *dst,
                                void *src,
                                void *stream) {

    auto &dstDesc = desc->dstDesc;
    auto &srcDesc = desc->srcDesc;

    aclTensor *td = dstDesc->t;
    aclTensor *ts = srcDesc->t;

    uint64_t workspaceSize;
    auto &handle = desc->handle;
    use_aclnn((AscendHandle_t) handle,
              [&](aclOpExecutor *&executor) {
                  auto ret = aclnnInplaceCopyGetWorkspaceSize(td,
                                                              ts,
                                                              &workspaceSize,
                                                              &executor);
                  aclSetAclOpExecutorRepeatable(executor);
                  CHECK_RET(ret == ACL_SUCCESS,
                            LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret));
              });
    desc->workspaceSize = workspaceSize;
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclrtMalloc failed, ERROR: %d\n", ret));
    }

    use_aclnn((AscendHandle_t) handle,
              [&](aclOpExecutor *executor) {
                  AclSetTensorAddr(executor, 0, td, dst);
                  AclSetTensorAddr(executor, 1, ts, src);
                  auto ret = aclnnInplaceCopy(workspaceAddr,
                                              desc->workspaceSize,
                                              executor,
                                              stream);
                  aclDestroyAclOpExecutor(executor);
                  CHECK_RET(ret == ACL_SUCCESS,
                            LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret));
              });

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnDestroyRearrangeDescriptor(RearrangeAclnnDescriptor_t desc) {
    delete desc->srcDesc;
    delete desc->dstDesc;
    delete desc;

    return STATUS_SUCCESS;
}
