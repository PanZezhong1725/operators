#ifndef __CUDA_SWIGLU_H__
#define __CUDA_SWIGLU_H__
#include "../../../devices/cuda/cuda_handle.h"
#include "../../../utils.h"
#include "operators.h"

struct SwiGLUCudaDescriptor {
    Device device;
    DT dtype;
    uint64_t seq_len;
    uint64_t di;
    uint64_t stride_a;
    uint64_t stride_b;
    uint64_t stride_c;
};

typedef struct SwiGLUCudaDescriptor *SwiGLUCudaDescriptor_t;

infiniopStatus_t cudaCreateSwiGLUDescriptor(CudaHandle_t handle,
                                            SwiGLUCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_dec,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc);

infiniopStatus_t cudaSwiGLU(SwiGLUCudaDescriptor_t desc,
                            void *c,
                            void const *a,
                            void const *b,
                            void *stream);

infiniopStatus_t cudaDestroySwiGLUDescriptor(SwiGLUCudaDescriptor_t desc);

void swiglu_nv_gpu_f16(SwiGLUCudaDescriptor_t desc, void *c, void const *a, void const *b, void *stream);

#endif// __NV_GPU_SWIGLU_H__
