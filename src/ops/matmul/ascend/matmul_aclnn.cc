#include "matmul_aclnn.h"

MatmulAclnnDescriptor::MatmulAclnnDescriptor(Device device) {
    this->device = device;
    handle = nullptr;
    cDesc = new aclnnTensorDescriptor();
    aDesc = new aclnnTensorDescriptor();
    bDesc = new aclnnTensorDescriptor();
    mt = 1;
    workspaceSize = 0;
}

infiniopStatus_t aclnnCreateMatmulDescriptor(AscendHandle_t handle,
                                             MatmulAclnnDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t c_desc,
                                             infiniopTensorDescriptor_t a_desc,
                                             infiniopTensorDescriptor_t b_desc,
                                             int8_t mt) {
    *desc_ptr = new MatmulAclnnDescriptor(handle->device);
    (*desc_ptr)->handle = handle;
    (*desc_ptr)->mt = mt;

    auto &cDesc = (*desc_ptr)->cDesc;
    auto &aDesc = (*desc_ptr)->aDesc;
    auto &bDesc = (*desc_ptr)->bDesc;

    auto status = cDesc->fromInfiniOpTensorDescriptor(c_desc);
    status = aDesc->fromInfiniOpTensorDescriptor(a_desc);
    status = bDesc->fromInfiniOpTensorDescriptor(b_desc);

    // printf("%s\n", cDesc->toString());
    // printf("%s\n", aDesc->toString());
    // printf("%s\n", bDesc->toString());

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

    use_aclnn_workspace((AscendHandle_t) handle,
                        [&](aclOpExecutor **executor) {
                            // auto ret =
                            //     aclnnBatchMatMulGetWorkspaceSize(ta, tb, tc, desc->mt, &workspaceSize, executor);
                            auto ret =
                                aclnnGemmGetWorkspaceSize(ta, tb, tc, 1.0, 0, transA, transB, tc,
                                                          desc->mt, &workspaceSize, executor);
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
                             float beta,
                             void const *a,
                             void const *b,
                             float alpha,
                             void *stream) {
    auto &cDesc = desc->cDesc;
    auto &aDesc = desc->aDesc;
    auto &bDesc = desc->bDesc;

    aclTensor *tc = cDesc->t;
    aclTensor *ta = aDesc->t;
    aclTensor *tb = bDesc->t;

    auto &handle = desc->handle;

    use_aclnn_compute(
        (AscendHandle_t) handle,
        [&](aclOpExecutor *&executor) {
            AclSetTensorAddr(executor, 0, ta, (void *) a);
            AclSetTensorAddr(executor, 1, tb, (void *) b);
            AclSetTensorAddr(executor, 2, tc, (void *) c);
            AclSetTensorAddr(executor, 3, tc, (void *) c);

            auto ret = aclnnGemm(workspace,
                                 desc->workspaceSize,
                                 executor,
                                 stream);
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


// void matmul_aclnn_f16(MatmulAclnnDescriptor *descriptor, Tensor c, float beta,
//                       Tensor a, Tensor b, float alpha, void *stream) {
//     // Copy tensor layout to descriptor
//     aclnnSetTensorDescriptorFromTensorLayout(descriptor->aDesc, a.layout);
//     aclnnSetTensorDescriptorFromTensorLayout(descriptor->bDesc, b.layout);
//     aclnnSetTensorDescriptorFromTensorLayout(descriptor->cDesc, c.layout);

//     // char *descStra = aclnnTensorDescToString(descriptor->aDesc);
//     // if (descStra) {
//     //     printf("%s", descStra);
//     // } else {
//     //     printf("Failed to print.\n");
//     // }

//     // char *descStrb = aclnnTensorDescToString(descriptor->bDesc);
//     // if (descStrb) {
//     //     printf("%s", descStrb);
//     // } else {
//     //     printf("Failed to print.\n");
//     // }

//     // char *descStrc = aclnnTensorDescToString(descriptor->cDesc);
//     // if (descStrc) {
//     //     printf("%s", descStrc);
//     // } else {
//     //     printf("Failed to print.\n");
//     // }

//     // Create aclTensor
//     aclTensor *self;
//     aclTensor *other;
//     aclTensor *out;

//     aclnnCreateTensor(descriptor->aDesc, a.data, &self);
//     aclnnCreateTensor(descriptor->bDesc, b.data, &other);
//     aclnnCreateTensor(descriptor->cDesc, c.data, &out);

//     // Get workspaceSize
//     uint64_t workspaceSize = 0;
//     aclOpExecutor *executor;
//     // TODO
//     int8_t cubeMathType = 1;
//     auto ret = aclnnBatchMatMulGetWorkspaceSize(self, other, out, cubeMathType,
//                                                 &workspaceSize, &executor);
//     CHECK_RET(ret == ACL_SUCCESS,
//               LOG_PRINT("aclnnBatchMatMulGetWorkspaceSize failed. ERROR: %d\n",
//                         ret););
//     // Malloc workspace on device
//     void *workspaceAddr = nullptr;
//     if (workspaceSize > 0) {
//         ret = aclrtMalloc(&workspaceAddr, workspaceSize,
//                           ACL_MEM_MALLOC_HUGE_FIRST);
//         CHECK_RET(ret == ACL_SUCCESS,
//                   LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret););
//     }
//     // Call aclnnBatchMatMul
//     ret = aclnnBatchMatMul(workspaceAddr, workspaceSize, executor, stream);
//     CHECK_RET(ret == ACL_SUCCESS,
//               LOG_PRINT("aclnnBatchMatMul failed. ERROR: %d\n", ret););

//     // Wait device work
//     ret = aclrtSynchronizeStream(stream);
//     CHECK_RET(ret == ACL_SUCCESS,
//               LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret););

//     aclDestroyTensor(self);
//     aclDestroyTensor(other);
//     aclDestroyTensor(out);

//     if (workspaceSize > 0) {
//         aclrtFree(workspaceAddr);
//     }

//     return;
// }
