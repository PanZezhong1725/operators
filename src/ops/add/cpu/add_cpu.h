#ifndef __CPU_ADD_H__
#define __CPU_ADD_H__

#include "operators.h"
#include <numeric>
#include <type_traits>

struct AddCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t c_data_size;
    uint64_t const *c_shape;
    uint64_t const *a_strides;
    uint64_t const *b_strides;
    uint64_t *c_indices;
};

typedef struct AddCpuDescriptor *AddCpuDescriptor_t;

infiniopStatus_t cpuCreateAddDescriptor(infiniopHandle_t,
                                        AddCpuDescriptor_t *,
                                        infiniopTensorDescriptor_t c,
                                        infiniopTensorDescriptor_t a,
                                        infiniopTensorDescriptor_t b);

infiniopStatus_t cpuAdd(AddCpuDescriptor_t desc,
                        void *c, void const *a, void const *b,
                        void *stream);

infiniopStatus_t cpuDestroyAddDescriptor(AddCpuDescriptor_t desc);

#endif
