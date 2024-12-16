#ifndef __ASCEND_RANDOM_SAMPLE_H__
#define __ASCEND_RANDOM_SAMPLE_H__

#include "../../../devices/ascend/ascend_handle.h"
#include "../../../devices/ascend/tensor_aclnn.h"
#include "../../utils.h"
#include "operators.h"
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <aclnnop/aclnn_topk.h>


struct RandomSampleAscendDescriptor {
    Device device;
    int device_id;
    aclnnTensorDescriptor_t pDesc;
    aclnnTensorDescriptor_t topkValDesc;
    aclnnTensorDescriptor_t topkIdxDesc;
    aclnnTensorDescriptor_t resDesc;
    RandomSampleAscendDescriptor(Device _device);
};

typedef struct RandomSampleAscendDescriptor *RandomSampleAscendDescriptor_t;

infiniopStatus_t ascendCreateRandomSampleDescriptor(AscendHandle_t handle,
                                                    RandomSampleAscendDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t result,
                                                    infiniopTensorDescriptor_t probs);

infiniopStatus_t ascendGetRandomSampleWorkspaceSize(RandomSampleAscendDescriptor_t desc,
                                                    uint64_t *size);

infiniopStatus_t ascendRandomSample(RandomSampleAscendDescriptor_t desc,
                                    void *workspace,
                                    uint64_t workspace_size,
                                    void *result,
                                    void const *probs,
                                    float random_val,
                                    float topp,
                                    int topk,
                                    float temperature,
                                    void *stream);

infiniopStatus_t ascendDestroyRandomSampleDescriptor(RandomSampleAscendDescriptor_t desc);

extern "C" infiniopStatus_t
random_sample_do(void *p, void *res, void *topkAddr, void *topkIdxAddr,
                 int32_t topk, int32_t voc, float topp, float temper,
                 float random, int dtype, void *stream);

#endif
