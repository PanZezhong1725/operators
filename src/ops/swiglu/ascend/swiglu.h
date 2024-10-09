#ifndef __ACLNN_SWIGLU_H__
#define __ACLNN_SWIGLU_H__

#include "operators.h"
#include "../../../utils.h"
#include <acl/acl_base.h>


struct SwiGLUAscendDescriptor {
    Device device;
    aclDataType dtype;
    int32_t seq_len;
    int32_t di;
    int32_t sta;
    int32_t stb;
    int32_t stc;
};

typedef struct SwiGLUAscendDescriptor *SwiGLUAscendDescriptor_t;

infiniopStatus_t ascendCreateSwiGLUDescriptor(infiniopHandle_t handle,
                                              SwiGLUAscendDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t c_desc,
                                              infiniopTensorDescriptor_t a_desc,
                                              infiniopTensorDescriptor_t b_desc);

infiniopStatus_t ascendSwiGLU(SwiGLUAscendDescriptor_t desc,
                              void *c,
                              void const *a,
                              void const *b,
                              void *stream);

infiniopStatus_t ascendDestroySwiGLUDescriptor(SwiGLUAscendDescriptor_t desc);

#endif