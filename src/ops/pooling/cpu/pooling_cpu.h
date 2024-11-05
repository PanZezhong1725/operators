#ifndef __CPU_POOLING_H__
#define __CPU_POOLING_H__

#include "../../../devices/cpu/common_cpu.h"
#include "operators.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

struct PoolingCpuDescriptor {
    Device device;
    DataLayout dt;
    uint64_t ndim;
    uint64_t y_size;
    uint64_t padded_x_size;
    uint64_t const *x_shape;
    uint64_t const *k_shape;
    uint64_t const *y_shape;
    uint64_t const *pads;
    int64_t const *strides;
    int pooling_mode;
};

typedef struct PoolingCpuDescriptor *PoolingCpuDescriptor_t;

infiniopStatus_t cpuCreatePoolingDescriptor(infiniopHandle_t handle,
                                            PoolingCpuDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x,
                                            void const *kernel_shape,
                                            void const *pads,
                                            void const *strides,
                                            uint64_t n,
                                            int pooling_type);

infiniopStatus_t cpuGetPoolingWorkspaceSize(PoolingCpuDescriptor_t desc, uint64_t *size);

infiniopStatus_t cpuPooling(PoolingCpuDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *y,
                            void const *x,
                            void *stream);

infiniopStatus_t cpuDestroyPoolingDescriptor(PoolingCpuDescriptor_t desc);

#endif
