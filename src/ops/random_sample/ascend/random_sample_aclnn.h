#ifndef __ASCEND_RANDOM_SAMPLE_H__
#define __ASCEND_RANDOM_SAMPLE_H__

#include "../../../devices/ascend/ascend_handle.h"
#include "../../../devices/ascend/tensor_aclnn.h"
#include "operators.h"
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <aclnnop/aclnn_argmax.h>


struct RandomSampleAclnnDescriptor {
    Device device;
    int device_id;
    aclOpExecutor *argMaxExecutor;
    aclnnTensorDescriptor_t pDesc;
    aclnnTensorDescriptor_t rDesc;
    float random_val;
    float topp;
    int topk;
    float temperature;
    uint64_t argMaxWorkspaceSize;
    RandomSampleAclnnDescriptor(Device _device);
};

typedef struct RandomSampleAclnnDescriptor *RandomSampleAclnnDescriptor_t;

infiniopStatus_t aclnnCreateRandomSampleDescriptor(AscendHandle_t handle,
                                                   RandomSampleAclnnDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t result,
                                                   infiniopTensorDescriptor_t probs);

infiniopStatus_t aclnnGetRandomSampleWorkspaceSize(RandomSampleAclnnDescriptor_t desc,
                                                   uint64_t *size);

infiniopStatus_t aclnnRandomSample(RandomSampleAclnnDescriptor_t desc,
                                   void *workspace,
                                   uint64_t workspace_size,
                                   void *result,
                                   void const *probs,
                                   float random_val,
                                   float topp,
                                   int topk,
                                   float temperature,
                                   void *stream);

infiniopStatus_t aclnnDestroyRandomSampleDescriptor(RandomSampleAclnnDescriptor_t desc);


#endif
