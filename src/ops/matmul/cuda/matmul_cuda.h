#ifndef __CUDA_MATMUL_H__
#define __CUDA_MATMUL_H__

#include "../blas.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <memory>

typedef struct MatmulCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    MatmulInfo info;
    std::shared_ptr<Pool<cublasHandle_t>> cublas_handles_t;
} MatmulCudaDescriptor;

typedef struct MatmulCudaDescriptor *MatmulCudaDescriptor_t;

infiniopStatus_t cudaCreateMatmulDescriptor(CudaHandle_t handle,
                                            MatmulCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc);

infiniopStatus_t cudaGetMatmulWorkspaceSize(MatmulCudaDescriptor_t desc, uint64_t *size);

infiniopStatus_t cudaMatmul(MatmulCudaDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *c,
                            float beta,
                            void const *a,
                            void const *b,
                            float alpha,
                            void *stream);

infiniopStatus_t cudaDestroyMatmulDescriptor(MatmulCudaDescriptor_t desc);

void matmul_cuda_f16(MatmulCudaDescriptor_t desc, void *c, float beta, void const *a, void const *b, float alpha, void *stream);

#endif// __CUDA_MATMUL_H__
