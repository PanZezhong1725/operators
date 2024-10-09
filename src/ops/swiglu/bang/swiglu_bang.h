#ifndef __BANG_SWIGLU_H__
#define __BANG_SWIGLU_H__

#include "../../../devices/bang/bang_handle.h"
#include "../../../utils.h"
#include "operators.h"

struct SwiGLUBangDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t seq_len;
    uint64_t di;
    uint64_t stride_a;
    uint64_t stride_b;
    uint64_t stride_c;
};

typedef struct SwiGLUBangDescriptor *SwiGLUBangDescriptor_t;

infiniopStatus_t bangCreateSwiGLUDescriptor(BangHandle_t handle,
                                            SwiGLUBangDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_dec,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc);

infiniopStatus_t bangSwiGLU(SwiGLUBangDescriptor_t desc,
                            void *c,
                            void const *a,
                            void const *b,
                            void *stream);

infiniopStatus_t bangDestroySwiGLUDescriptor(SwiGLUBangDescriptor_t desc);

#endif// __BANG_SWIGLU_H__
