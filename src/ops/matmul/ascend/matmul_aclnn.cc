#include "matmul_aclnn.h"

MatmulAclnnDescriptor::MatmulAclnnDescriptor(Device device) {
    this->device = device;
    handle = nullptr;
    cDesc = new aclnnTensorDescriptor();
    aDesc = new aclnnTensorDescriptor();
    bDesc = new aclnnTensorDescriptor();
    alpha = 1.0;
    beta = 0;
    mt = 1;
    workspaceSize = 0;
}

infiniopStatus_t aclnnCreateMatmulDescriptor(AscendHandle_t handle,
                                             MatmulAclnnDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t c_desc,
                                             float alpha,
                                             infiniopTensorDescriptor_t a_desc,
                                             infiniopTensorDescriptor_t b_desc,
                                             float beta,
                                             int8_t mt) {
    *desc_ptr = new MatmulAclnnDescriptor(handle->device);
    (*desc_ptr)->handle = handle;
    (*desc_ptr)->mt = mt;
    (*desc_ptr)->alpha = alpha;
    (*desc_ptr)->beta = beta;

    auto &cDesc = (*desc_ptr)->cDesc;
    auto &aDesc = (*desc_ptr)->aDesc;
    auto &bDesc = (*desc_ptr)->bDesc;

    auto status = cDesc->fromInfiniOpTensorDescriptor(c_desc);
    status = aDesc->fromInfiniOpTensorDescriptor(a_desc);
    status = bDesc->fromInfiniOpTensorDescriptor(b_desc);

    status = cDesc->createTensor();
    status = aDesc->createTensor();
    status = bDesc->createTensor();

    return status;
}

infiniopStatus_t aclnnGetMatmulWorkspaceSize(MatmulAclnnDescriptor_t desc,
                                             uint64_t *size) {
    auto &cDesc = desc->cDesc;
    auto &aDesc = desc->aDesc;
    auto &bDesc = desc->bDesc;

    aclTensor *tc = cDesc->t;
    aclTensor *ta = aDesc->t;
    aclTensor *tb = bDesc->t;

    uint64_t workspaceSize;
    auto &handle = desc->handle;

    // Get transA and transB according strides
    int64_t transA = aDesc->strides[aDesc->ndim - 1] == 1 ? 0 : 1;
    int64_t transB = bDesc->strides[bDesc->ndim - 1] == 1 ? 0 : 1;

    use_aclnn((AscendHandle_t) handle,
              [&](aclOpExecutor *&executor) {
                  // auto ret =
                  //     aclnnBatchMatMulGetWorkspaceSize(ta, tb, tc, desc->mt, &workspaceSize, executor);
                  auto ret =
                      aclnnGemmGetWorkspaceSize(ta, tb, tc, desc->alpha, desc->beta, transA, transB, tc,
                                                desc->mt, &workspaceSize, &executor);
                  aclSetAclOpExecutorRepeatable(executor);
                  CHECK_RET(ret == ACL_SUCCESS,
                            LOG_PRINT("aclnnGemmGetWorkspaceSize failed. ERROR: %d\n", ret));
                  // printf("%s\n", aclGetRecentErrMsg());
              });
    *size = workspaceSize;
    desc->workspaceSize = workspaceSize;

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnMatmul(MatmulAclnnDescriptor_t desc,
                             void *workspace,
                             uint64_t workspace_size,
                             void *c,
                             void const *a,
                             void const *b,
                             void *stream) {
    auto &cDesc = desc->cDesc;
    auto &aDesc = desc->aDesc;
    auto &bDesc = desc->bDesc;

    aclTensor *tc = cDesc->t;
    aclTensor *ta = aDesc->t;
    aclTensor *tb = bDesc->t;

    auto &handle = desc->handle;

    use_aclnn(
        (AscendHandle_t) handle,
        [&](aclOpExecutor *executor) {
            AclSetTensorAddr(executor, 0, ta, (void *) a);
            AclSetTensorAddr(executor, 1, tb, (void *) b);
            AclSetTensorAddr(executor, 2, tc, (void *) c);
            AclSetTensorAddr(executor, 3, tc, (void *) c);

            auto ret = aclnnGemm(workspace,
                                 desc->workspaceSize,
                                 executor,
                                 stream);
            aclDestroyAclOpExecutor(executor);
            CHECK_RET(ret == ACL_SUCCESS,
                      LOG_PRINT("aclnnBatchMatMul failed. ERROR: %d\n", ret));
        });

    return STATUS_SUCCESS;
}


infiniopStatus_t aclnnDestroyMatmulDescriptor(MatmulAclnnDescriptor_t desc) {
    delete desc->cDesc;
    delete desc->bDesc;
    delete desc->aDesc;

    return STATUS_SUCCESS;
}
