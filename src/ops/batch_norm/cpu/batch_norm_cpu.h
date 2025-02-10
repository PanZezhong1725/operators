#ifndef __CPU_BATCH_NORM_H__
#define __CPU_BATCH_NORM_H__

#include "operators.h"
#include <cmath>
#include <numeric>

struct BatchNormCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t batch_size;
    uint64_t channel_size;
    uint64_t spatial_data_size;
    uint64_t per_batch_data_size;
    double eps;
};

typedef struct BatchNormCpuDescriptor *BatchNormCpuDescriptor_t;

infiniopStatus_t cpuCreateBatchNormDescriptor(infiniopHandle_t,
                                              BatchNormCpuDescriptor_t *,
                                              infiniopTensorDescriptor_t y,
                                              infiniopTensorDescriptor_t x,
                                              infiniopTensorDescriptor_t scale,
                                              infiniopTensorDescriptor_t b,
                                              infiniopTensorDescriptor_t mean,
                                              infiniopTensorDescriptor_t var,
                                              double eps);

infiniopStatus_t cpuBatchNorm(BatchNormCpuDescriptor_t desc,
                              void *y, void const *x, void const *scale, void const *b,
                              void const *mean, void const *var, void *stream);

infiniopStatus_t cpuDestroyBatchNormDescriptor(BatchNormCpuDescriptor_t desc);

#endif
