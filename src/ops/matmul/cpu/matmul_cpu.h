#ifndef __CPU_MATMUL_H__
#define __CPU_MATMUL_H__

#include "../../../devices/cpu/cpu_handle.h"
#include "../blas.h"
#include "operators.h"

typedef struct MatmulCpuDescriptor {
    Device device;
    DT dtype;
    MatmulInfo info;
} MatmulCpuDescriptor;

typedef struct MatmulCpuDescriptor *MatmulCpuDescriptor_t;

infiniopStatus_t cpuCreateMatmulDescriptor(CpuHandle_t handle,
                                           MatmulCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t c_desc,
                                           infiniopTensorDescriptor_t a_desc,
                                           infiniopTensorDescriptor_t b_desc);

infiniopStatus_t cpuGetMatmulWorkspaceSize(MatmulCpuDescriptor_t desc, uint64_t *size);

infiniopStatus_t cpuMatmul(MatmulCpuDescriptor_t desc,
                           void *workspace,
                           uint64_t workspace_size,
                           void *c,
                           float beta,
                           void const *a,
                           void const *b,
                           float alpha);

infiniopStatus_t cpuDestroyMatmulDescriptor(MatmulCpuDescriptor_t desc);

void matmul_cpu_f16(MatmulCpuDescriptor_t desc, void *c, float beta, void const *a, void const *b, float alpha);

#endif// __CPU_MATMUL_H__
