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

// get the total number of elements in arr
inline uint64_t getTotalSize(const uint64_t *arr, uint64_t ndim) {
    return std::accumulate(arr, arr + ndim, 1ULL, std::multiplies<uint64_t>());
}

// check if padding is needed
inline bool requirePadding(uint64_t const *pads, uint64_t ndim) {
    return std::any_of(pads, pads + ndim - 2,
                       [](uint64_t pad) { return pad > 0; });
}

/**
 * get the total array size (element count) after applying padding for a
 * ndim-ary tensor with the given shape
 */
uint64_t getPaddedSize(uint64_t ndim, uint64_t *shape, uint64_t const *pads);

// calculate the padded shape and store the result in padded_shape
void getPaddedShape(uint64_t ndim, uint64_t const *shape, uint64_t const *pads, uint64_t *padded_shape);

// copy the data in src tensor into that of the dest tensor but also convert
// from f32 to f16
inline void copyF32DataToF16(uint16_t *dest, float const *src, uint64_t size) {
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        dest[i] = f32_to_f16(src[i]);
    }
}

#endif
