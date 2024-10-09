#include "swiglu_aclnn.h"
#include "../../../utils.h"

SwiGLUAclnnDescriptor::SwiGLUAclnnDescriptor(Device _device) {
    device = _device;
    handle = nullptr;
    cDesc = new aclnnTensorDescriptor();
    aDesc = new aclnnTensorDescriptor();
    bDesc = new aclnnTensorDescriptor();
    catDesc = new aclnnTensorDescriptor();
}

infiniopStatus_t aclnnCreateSwiGLUDescriptor(AscendHandle_t handle,
                                             SwiGLUAclnnDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t c_desc,
                                             infiniopTensorDescriptor_t a_desc,
                                             infiniopTensorDescriptor_t b_desc) {
    *desc_ptr = new SwiGLUAclnnDescriptor(handle->device);
    (*desc_ptr)->handle = reinterpret_cast<AscendHandle_t>(handle);

    auto &cDesc = (*desc_ptr)->cDesc;
    auto &aDesc = (*desc_ptr)->aDesc;
    auto &bDesc = (*desc_ptr)->bDesc;
    auto &catDesc = (*desc_ptr)->catDesc;


    auto status = cDesc->fromInfiniOpTensorDescriptor(c_desc);
    status = aDesc->fromInfiniOpTensorDescriptor(a_desc);
    status = bDesc->fromInfiniOpTensorDescriptor(b_desc);

    // Create catDesc
    auto ndim = cDesc->ndim;
    auto cat_shape = new std::vector<int64_t>(ndim);
    auto cat_strides = new std::vector<int64_t>(ndim, 1);

    // Infer shape of Cat[a, b] at last dim
    for (uint64_t i = 0; i < ndim; ++i) {
        (*cat_shape)[i] = cDesc->shape[i];
    }
    (*cat_shape)[ndim - 1] = aDesc->shape[ndim - 1] + bDesc->shape[ndim - 1];

    // Infer Continious tensor strides
    for (auto i = static_cast<int64_t>(ndim - 2); i >= 0; --i) {
        (*cat_strides)[i] = (*cat_shape)[i + 1] * (*cat_strides)[i + 1];
    }

    // Set catTensor descriptor
    catDesc->ndim = cDesc->ndim;
    catDesc->shape = (*cat_shape).data();
    catDesc->strides = (*cat_strides).data();
    catDesc->offset = 0;
    catDesc->dataType = cDesc->dataType;
    catDesc->format = cDesc->format;
    catDesc->storageShape = (*cat_shape).data();
    catDesc->storageNdim = cDesc->ndim;

    status = aDesc->createTensor();
    status = bDesc->createTensor();
    status = cDesc->createTensor();
    status = catDesc->createTensor();

    return status;
}

infiniopStatus_t aclnnSwiGLU(SwiGLUAclnnDescriptor_t desc,
                             void *c,
                             void const *a,
                             void const *b,
                             void *stream) {
    auto &cDesc = desc->cDesc;
    auto &aDesc = desc->aDesc;
    auto &bDesc = desc->bDesc;
    auto &catDesc = desc->catDesc;

    auto &handle = desc->handle;

    // Get Tensor
    aclTensor *tc = cDesc->t;
    aclTensor *ta = aDesc->t;
    aclTensor *tb = bDesc->t;
    aclTensor *tcat = catDesc->t;

    // Malloc catDesc data
    void *cat_data = nullptr;
    auto ret = aclrtMalloc(&cat_data,
                           numElements(catDesc->shape, catDesc->ndim) * aclDataTypeSize(catDesc->dataType),
                           ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret));

    std::vector<aclTensor *> tmp{tb, ta};
    aclTensorList *tensorList = aclCreateTensorList(tmp.data(), tmp.size());

    // aclnnCat
    uint64_t workspaceSize = 0;
    aclOpExecutor *tmpExecutor;
    ret = aclnnCatGetWorkspaceSize(tensorList, -1, tcat, &workspaceSize,
                                   &tmpExecutor);
    aclSetAclOpExecutorRepeatable(tmpExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnCatGetWorkspaceSize failed. ERROR: %d\n", ret));

    // Set tensor addr
    AclSetTensorAddr(tmpExecutor, 0, tb, (void *) b);
    AclSetTensorAddr(tmpExecutor, 1, ta, (void *) a);
    AclSetTensorAddr(tmpExecutor, 2, tcat, (void *) cat_data);

    // Malloc workspace on device
    void *workspaceAddr = mallocWorkspace(workspaceSize);

    ret = aclnnCat(workspaceAddr, workspaceSize, tmpExecutor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnCat failed. ERROR: %d\n", ret));
    freeWorkspace(workspaceAddr);
    aclDestroyAclOpExecutor(tmpExecutor);


    // Cat a,b
    use_aclnn((AscendHandle_t) handle,
              [&](aclOpExecutor *&executor) {
                  // aclnnSwiGLU
                  ret = aclnnSwiGluGetWorkspaceSize(tcat,
                                                    -1,
                                                    tc,
                                                    &workspaceSize,
                                                    &executor);
                  aclSetAclOpExecutorRepeatable(executor);
                  CHECK_RET(ret == ACL_SUCCESS,
                            LOG_PRINT("aclnnSwiGluGetWorkspaceSize failed. ERROR: %d\n", ret));

                  AclSetTensorAddr(executor, 0, tcat, (void *) cat_data);
                  AclSetTensorAddr(executor, 1, tc, (void *) c);

                  uint64_t workspaceSize = 0;
                  void *workspaceAddr = mallocWorkspace(workspaceSize);

                  CHECK_RET(ret == ACL_SUCCESS,
                            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret));

                  ret = aclnnSwiGlu(workspaceAddr, workspaceSize, executor, stream);
                  CHECK_RET(ret == ACL_SUCCESS,
                            LOG_PRINT("aclnnSwiGlu failed. ERROR: %d\n", ret));
                 
                  freeWorkspace(workspaceAddr);
                  aclDestroyAclOpExecutor(executor);
              });

    aclrtFree(cat_data);
    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnDestroySwiGLUDescriptor(SwiGLUAclnnDescriptor_t desc) {
    delete desc->cDesc;
    delete desc->aDesc;
    delete desc->bDesc;
    delete desc->catDesc;

    return STATUS_SUCCESS;
}
