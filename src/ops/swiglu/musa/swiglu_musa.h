#ifndef __MUSA_SWIGLU_H__
#define __MUSA_SWIGLU_H__

#include "operators.h"

struct SwiGLUMusaDescriptor {
    Device device;
    DT dtype;
    uint64_t seq_len;
    uint64_t di;
    uint64_t stride_a;
    uint64_t stride_b;
    uint64_t stride_c;
};

typedef struct SwiGLUMusaDescriptor *SwiGLUMusaDescriptor_t;

infiniopStatus_t musaCreateSwiGLUDescriptor(infiniopHandle_t handle,
                                            SwiGLUMusaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_dec,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc);

infiniopStatus_t musaSwiGLU(SwiGLUMusaDescriptor_t desc,
                            void *c,
                            void *a,
                            void *b,
                            void *stream);

infiniopStatus_t musaDestroySwiGLUDescriptor(SwiGLUMusaDescriptor_t desc);

void swiglu_mt_gpu_f16(SwiGLUMusaDescriptor_t desc, void *c, void *a, void *b, void *stream);

#endif// __MUSA_GPU_SWIGLU_H__
