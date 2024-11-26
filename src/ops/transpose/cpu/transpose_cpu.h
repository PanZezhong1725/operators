#ifndef __CPU_TRANSPOSE_H__
#define __CPU_TRANSPOSE_H__

#include "../transpose_common.h"
#include <cstring>
#include <numeric>

struct TransposeCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t data_size;
    int64_t const *x_strides;
    int64_t const *y_strides;
    uint64_t const *x_shape;
    uint64_t const *y_shape;
    TransposeMode mode;
};

typedef struct TransposeCpuDescriptor *TransposeCpuDescriptor_t;


infiniopStatus_t cpuCreateTransposeDescriptor(infiniopHandle_t,
                                              TransposeCpuDescriptor_t *,
                                              infiniopTensorDescriptor_t y,
                                              infiniopTensorDescriptor_t x,
                                              uint64_t const *perm,
                                              uint64_t n);

infiniopStatus_t cpuTranspose(TransposeCpuDescriptor_t desc,
                              void *y, void const *x,
                              void *stream);

infiniopStatus_t cpuDestroyTransposeDescriptor(TransposeCpuDescriptor_t desc);

#endif
