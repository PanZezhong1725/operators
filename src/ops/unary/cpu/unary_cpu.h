#ifndef __CPU_UNARY_H__
#define __CPU_UNARY_H__

#include "../unary.h"
#include "operators.h"
#include <numeric>
#include <type_traits>

struct UnaryCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t data_size;
    int mode;
};

typedef struct UnaryCpuDescriptor *UnaryCpuDescriptor_t;

infiniopStatus_t cpuCreateUnaryDescriptor(infiniopHandle_t,
                                          UnaryCpuDescriptor_t *,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x,
                                          int mode);

infiniopStatus_t cpuUnary(UnaryCpuDescriptor_t desc,
                          void *y, void const *x,
                          void *stream);

infiniopStatus_t cpuDestroyUnaryDescriptor(UnaryCpuDescriptor_t desc);

#endif