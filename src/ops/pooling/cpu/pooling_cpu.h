#ifndef __CPU_POOLING_H__
#define __CPU_POOLING_H__

#include "operators.h"
struct PoolingCpuDescriptor {
    Device device;
    DataLayout dt;
    uint64_t ndim;
    // uint64_t y_size;
    // uint64_t padded_x_size;
    // uint64_t const *x_shape;
    // uint64_t const *w_shape;
    // uint64_t const *y_shape;
    // uint64_t const *pads;
    // int64_t const *strides;
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

infiniopStatus_t cpuPooling(PoolingCpuDescriptor_t desc,
                            void *y,
                            void const *x,
                            void *stream);

infiniopStatus_t cpuDestroyPoolingDescriptor(PoolingCpuDescriptor_t desc);

#endif
