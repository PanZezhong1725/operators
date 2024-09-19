#ifndef __ACLNN_MATMUL_H__
#define __ACLNN_MATMUL_H__

#include "../../../devices/ascend/ascend_handle.h"
#include "../../../devices/ascend/tensor_aclnn.h"
#include "operators.h"
#include "../../../utils.h"
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/level2/aclnn_gemm.h>

struct MatmulAclnnDescriptor {
    Device device;
    AscendHandle_t handle;
    aclnnTensorDescriptor_t cDesc, aDesc, bDesc;
    // cubeMathType
    // see doc: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/apiref/appdevgapi/context/aclnnBatchMatMul.md
    int8_t mt;
    uint64_t workspaceSize;

    MatmulAclnnDescriptor(Device device);
};

typedef struct MatmulAclnnDescriptor *MatmulAclnnDescriptor_t;

infiniopStatus_t aclnnCreateMatmulDescriptor(AscendHandle_t handle,
                                             MatmulAclnnDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t c_desc,
                                             infiniopTensorDescriptor_t a_desc,
                                             infiniopTensorDescriptor_t b_desc,
                                             int8_t cubeMathType);

infiniopStatus_t aclnnGetMatmulWorkspaceSize(MatmulAclnnDescriptor_t desc,
                                             uint64_t *size);

infiniopStatus_t aclnnMatmul(MatmulAclnnDescriptor_t desc,
                             void *workspace,
                             uint64_t workspace_size,
                             void *c,
                             float beta,
                             const void *a,
                             const void *b,
                             float alpha,
                             void *stream);

infiniopStatus_t aclnnDestroyMatmulDescriptor(MatmulAclnnDescriptor_t desc);

#endif