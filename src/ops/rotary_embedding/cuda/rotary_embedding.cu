#include "../../utils.h"
#include "rotary_embedding.cuh"
#include <cuda_fp16.h>

static __global__ void padding(
    half2 *__restrict__ x_,
    unsigned long const *__restrict__ pos_,
    float const *__restrict__ sin_,
    float const *__restrict__ cos_,
    long const stride0,
    long const stride1) {
    auto dk = blockDim.x;
    auto k = threadIdx.x;
    auto offset = blockIdx.x * stride0 + blockIdx.y * stride1 + k;
    auto &x = x_[offset];
    auto pos = pos_[blockIdx.x];
    auto sincos_offset = pos * dk * 2 + k * 2;

    float sin0 = sin_[sincos_offset], cos0 = cos_[sincos_offset],
          sin1 = sin_[sincos_offset + 1], cos1 = cos_[sincos_offset + 1];
    float x0 = __half2float(x.x) * cos0 - __half2float(x.y) * sin0;
    float x1 = __half2float(x.y) * cos1 + __half2float(x.x) * sin1;
    x = half2(x0, x1);
}


void rotary_embedding_nv_gpu_f16(
    RoPECudaDescriptor_t desc,
    half2 *t,
    unsigned long const *pos,
    float const *sin_, float const *cos_,
    void *stream) {
    auto nt = desc->seq_len,
         nh = desc->nhead,
         dh = desc->dim;

    // batching 2 half together
    auto stride0 = desc->strides[0] / 2,
         stride1 = desc->strides[1] / 2;

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    padding<<<dim3(nt, nh), dh / 2, 0, cuda_stream>>>(t, pos, sin_, cos_, stride0, stride1);
}

infiniopStatus_t cudaRoPE(RoPECudaDescriptor_t desc,
                          void *workspace,
                          unsigned long int workspace_size,
                          void *t,
                          void const *pos_ids,
                          void const *sin_table,
                          void const *cos_table,
                          void *stream) {
    if (t == nullptr || pos_ids == nullptr || sin_table == nullptr || cos_table == nullptr)
        return STATUS_BAD_PARAM;

    if (dtype_eq(desc->dtype, F16)) {
        rotary_embedding_nv_gpu_f16(desc,
                                    reinterpret_cast<half2 *>(t),
                                    reinterpret_cast<unsigned long const *>(pos_ids),
                                    reinterpret_cast<float const *>(sin_table),
                                    reinterpret_cast<float const *>(cos_table),
                                    stream);
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    return STATUS_SUCCESS;
}
