#ifndef __CPU_RELU_H__
#define __CPU_RELU_H__

#include "operators.h"
#include <numeric>

struct ReluCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t data_size;
};

typedef struct ReluCpuDescriptor *ReluCpuDescriptor_t;

infiniopStatus_t cpuCreateReluDescriptor(infiniopHandle_t,
                                         ReluCpuDescriptor_t *,
                                         infiniopTensorDescriptor_t y,
                                         infiniopTensorDescriptor_t x);

infiniopStatus_t cpuRelu(ReluCpuDescriptor_t desc,
                         void *y, void const *x,
                         void *stream);

infiniopStatus_t cpuDestroyReluDescriptor(ReluCpuDescriptor_t desc);

#endif
