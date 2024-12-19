#ifndef __CPU_REARRANGE_H__
#define __CPU_REARRANGE_H__

#include "operators.h"
#include <vector>
struct RearrangeCpuDescriptor {
    Device device;
    DataLayout dt;
    uint64_t r;
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides_dst;
    std::vector<int64_t> strides_src;
};

typedef struct RearrangeCpuDescriptor *RearrangeCpuDescriptor_t;

infiniopStatus_t cpuCreateRearrangeDescriptor(infiniopHandle_t handle,
                                              RearrangeCpuDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t dst,
                                              infiniopTensorDescriptor_t src);

infiniopStatus_t cpuRearrange(RearrangeCpuDescriptor_t desc,
                              void *dst,
                              void const *src,
                              void *stream);

infiniopStatus_t cpuDestroyRearrangeDescriptor(RearrangeCpuDescriptor_t desc);

void reform_cpu(RearrangeCpuDescriptor_t desc, void *y, void const *x);

#endif
