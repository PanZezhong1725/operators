#ifndef __CUDA_MATMUL_H__
#define __CUDA_MATMUL_H__

#include "../../../../include/operators.h"
#include "../../../devices/iluvatar/ilu_handle.h"
#include "../blas.h"
#include <memory>

typedef struct MatmulIluDescriptor {
    Device device;
    DT dtype;
    int device_id;
    MatmulInfo info;
    float alpha;
    float beta;
    std::shared_ptr<Pool<cublasHandle_t>> cublas_handles_t;
} MatmulIluDescriptor;

typedef struct MatmulIluDescriptor *MatmulIluDescriptor_t;

infiniopStatus_t iluCreateMatmulDescriptor(IluHandle_t handle,
                                           MatmulIluDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t c_desc,
                                           float alpha,
                                           infiniopTensorDescriptor_t a_desc,
                                           infiniopTensorDescriptor_t b_desc,
                                           float beta);

infiniopStatus_t iluGetMatmulWorkspaceSize(MatmulIluDescriptor_t desc, uint64_t *size);

infiniopStatus_t iluMatmul(MatmulIluDescriptor_t desc,
                           void *workspace,
                           uint64_t workspace_size,
                           void *c,
                           void const *a,
                           void const *b,
                           void *stream);

infiniopStatus_t iluDestroyMatmulDescriptor(MatmulIluDescriptor_t desc);

#endif// __ILU_MATMUL_H__
