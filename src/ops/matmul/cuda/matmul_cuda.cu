#include "../../../devices/cuda/handle_pool.h"
#include "../../utils.h"
#include "../blas.h"
#include "matmul_cuda.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>

void matmul_cuda_f16(MatmulCudaDescriptor_t desc, void *c, float beta, void const *a, void const *b, float alpha, void *stream) {
    auto info = desc->info;

    if (info.is_transed) {
        std::swap(a, b);
    }

    auto alpha_f16 = __float2half(alpha);
    auto beta_f16 = __float2half(beta);

    auto op_a = info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

    use_cublas((cudaStream_t) stream,
               [&](cublasHandle_t handle) { cublasGemmStridedBatchedEx(
                                                handle,
                                                op_a,
                                                op_b,
                                                info.m,
                                                info.n,
                                                info.k,
                                                &alpha_f16,
                                                a,
                                                CUDA_R_16F,
                                                info.a_matrix.ld(),
                                                info.a_matrix.stride,
                                                b,
                                                CUDA_R_16F,
                                                info.b_matrix.ld(),
                                                info.b_matrix.stride,
                                                &beta_f16,
                                                c,
                                                CUDA_R_16F,
                                                info.c_matrix.ld(),
                                                info.c_matrix.stride,
                                                info.batch,
                                                CUBLAS_COMPUTE_16F,
                                                CUBLAS_GEMM_DEFAULT_TENSOR_OP); });
}
