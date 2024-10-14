#include "../../../devices/cuda/common_cuda.h"
#include "../../../utils.h"
#include "swiglu.cuh"
#include <cuda_fp16.h>

static __forceinline__ __device__ float silu(float x) {
    return x * fdividef(1, 1 + expf(-x));
}

inline int gcd(int a, int b) {
    while (b != 0) {
        int rem = a % b;
        a = b;
        b = rem;
    }
    return a;
}

template<class Tdata>
static __global__ void swiglu(
    Tdata *__restrict__ c,
    int const stride_c,
    Tdata const *__restrict__ a,
    int const stride_a,
    Tdata const *__restrict__ b,
    int const stride_b) {
    auto i = blockIdx.y * stride_b + blockIdx.x * blockDim.x + threadIdx.x,
         j = blockIdx.y * stride_a + blockIdx.x * blockDim.x + threadIdx.x,
         k = blockIdx.y * stride_c + blockIdx.x * blockDim.x + threadIdx.x;
    auto x = float(b[i]),
         y = float(a[j]);
    c[k] = Tdata(silu(x) * y);
}

void swiglu_nv_gpu_f16(SwiGLUCudaDescriptor_t desc, void *c, void const *a, void const *b, void *stream) {

    auto seq_len = desc->seq_len,
         di = desc->di;

    auto stride_a = desc->stride_a,
         stride_b = desc->stride_b,
         stride_c = desc->stride_c;

    dim3 block_dims = gcd(MAX_THREADS_PER_BLOCK, di);
    dim3 grid_dims = dim3(di / block_dims.x, seq_len);

    auto a_ptr = reinterpret_cast<const half *>(a);
    auto b_ptr = reinterpret_cast<const half *>(b);
    auto c_ptr = reinterpret_cast<half *>(c);

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    swiglu<<<grid_dims, block_dims, 0, cuda_stream>>>(
        c_ptr, stride_c, a_ptr, stride_a, b_ptr, stride_b);
}

infiniopStatus_t cudaSwiGLU(SwiGLUCudaDescriptor_t desc,
                            void *c,
                            void const *a,
                            void const *b,
                            void *stream) {
    if (dtype_eq(desc->dtype, F16)) {
        swiglu_nv_gpu_f16(desc, c, a, b, stream);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
