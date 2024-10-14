#ifndef __ACLNN_SWIGLU_H__
#define __ACLNN_SWIGLU_H__

#include "../../../devices/ascend/ascend_handle.h"
#include "../../../devices/ascend/tensor_aclnn.h"
#include "../../../devices/ascend/common_ascend.h"
#include "operators.h"
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_cat.h>
#include <aclnnop/aclnn_swi_glu.h>

struct SwiGLUAclnnDescriptor {
    Device device;
    AscendHandle_t handle;
    aclnnTensorDescriptor_t cDesc, aDesc, bDesc;
    aclnnTensorDescriptor_t catDesc;

    SwiGLUAclnnDescriptor(Device device);
};

typedef SwiGLUAclnnDescriptor *SwiGLUAclnnDescriptor_t;

infiniopStatus_t aclnnCreateSwiGLUDescriptor(AscendHandle_t handle,
                                             SwiGLUAclnnDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t c_desc,
                                             infiniopTensorDescriptor_t a_desc,
                                             infiniopTensorDescriptor_t b_desc);

infiniopStatus_t aclnnSwiGLU(SwiGLUAclnnDescriptor_t desc,
                             void *c,
                             void const *a,
                             void const *b,
                             void *stream);

infiniopStatus_t aclnnDestroySwiGLUDescriptor(SwiGLUAclnnDescriptor_t desc);

#endif
