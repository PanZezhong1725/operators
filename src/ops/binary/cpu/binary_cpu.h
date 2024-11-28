#ifndef __CPU_BINARY_H__
#define __CPU_BINARY_H__

#include "../binary.h"
#include "operators.h"
#include <numeric>
#include <type_traits>

struct BinaryCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t c_data_size;
    uint64_t const *c_shape;
    int64_t const *a_strides;
    int64_t const *b_strides;
    int64_t const *c_strides;
    bool broadcasted;
    int mode;
};

typedef struct BinaryCpuDescriptor *BinaryCpuDescriptor_t;

infiniopStatus_t cpuCreateBinaryDescriptor(infiniopHandle_t,
                                           BinaryCpuDescriptor_t *,
                                           infiniopTensorDescriptor_t c,
                                           infiniopTensorDescriptor_t a,
                                           infiniopTensorDescriptor_t b,
                                           int mode);

infiniopStatus_t cpuBinary(BinaryCpuDescriptor_t desc,
                           void *c, void const *a, void const *b,
                           void *stream);

infiniopStatus_t cpuDestroyBinaryDescriptor(BinaryCpuDescriptor_t desc);

#endif
