#include "random_sample_aclnn.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

RandomSampleAclnnDescriptor::RandomSampleAclnnDescriptor(Device _device) {
    device = _device;
    device_id = 0;
    argMaxExecutor = nullptr;
    pDesc = new aclnnTensorDescriptor();
    rDesc = new aclnnTensorDescriptor();
    random_val = 1.0;
    topp = 0;
    topk = 0;
    temperature = 1.0;
    argMaxWorkspaceSize = 0;
}

infiniopStatus_t aclnnCreateRandomSampleDescriptor(AscendHandle_t handle,
                                                   RandomSampleAclnnDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t result,
                                                   infiniopTensorDescriptor_t probs) {

    (*desc_ptr) = new RandomSampleAclnnDescriptor(handle->device);
    (*desc_ptr)->device_id = handle->device_id;
    (*desc_ptr)->random_val = 0;
    (*desc_ptr)->topp = 0;
    (*desc_ptr)->topk = 0;
    (*desc_ptr)->temperature = 1.0;

    auto &pDesc = (*desc_ptr)->pDesc;
    auto &rDesc = (*desc_ptr)->rDesc;

    CHECK_STATUS(pDesc->fromInfiniOpTensorDescriptor(probs), STATUS_SUCCESS);
    CHECK_STATUS(pDesc->createTensor(), STATUS_SUCCESS);

    result->dt = I64;
    CHECK_STATUS(rDesc->fromInfiniOpTensorDescriptor(result), STATUS_SUCCESS);
    CHECK_STATUS(rDesc->createTensor(), STATUS_SUCCESS);

    aclTensor *tp = pDesc->t;
    aclTensor *tr = rDesc->t;

    aclnnStatus ret;

    // temp = prob / temperature
    auto &argmaxWorkspaceSize = (*desc_ptr)->argMaxWorkspaceSize;
    auto &argmaxExecutor = (*desc_ptr)->argMaxExecutor;
    ret = aclnnArgMaxGetWorkspaceSize(tp,
                                      0,
                                      true,
                                      tr,
                                      &argmaxWorkspaceSize,
                                      &argmaxExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnArgMaxGetWorkspaceSize failed, ERROR: %d\n", ret);
              return STATUS_EXECUTION_FAILED);
    aclSetAclOpExecutorRepeatable(argmaxExecutor);
    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnGetRandomSampleWorkspaceSize(RandomSampleAclnnDescriptor_t desc, uint64_t *size) {
    *size = desc->argMaxWorkspaceSize;
    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnRandomSample(RandomSampleAclnnDescriptor_t desc,
                                   void *workspace,
                                   uint64_t workspace_size,
                                   void *result,
                                   void const *probs,
                                   float random_val,
                                   float topp,
                                   int topk,
                                   float temperature,
                                   void *stream) {
    auto &pDesc = desc->pDesc;
    auto &rDesc = desc->rDesc;

    aclTensor *tp = pDesc->t;
    aclTensor *tr = rDesc->t;

    aclrtSetDevice(desc->device_id);

    auto &argmaxWorkspaceSize = desc->argMaxWorkspaceSize;
    auto &argmaxExecutor = desc->argMaxExecutor;

    AclSetTensorAddr(argmaxExecutor, 0, tp, (void *) probs);
    AclSetTensorAddr(argmaxExecutor, 1, tr, (void *) result);
    auto ret = aclnnArgMax(workspace,
                           argmaxWorkspaceSize,
                           argmaxExecutor,
                           stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnArgMax failed. ERROR: %d\n", ret);
              return STATUS_EXECUTION_FAILED);
    return STATUS_SUCCESS;
}


infiniopStatus_t aclnnDestroyRandomSampleDescriptor(RandomSampleAclnnDescriptor_t desc) {
    delete desc->pDesc;
    delete desc->rDesc;
    aclDestroyAclOpExecutor(desc->argMaxExecutor);
    delete desc;

    return STATUS_SUCCESS;
}
