#ifndef __CPU_SOFTMAX_H__
#define __CPU_SOFTMAX_H__

#include "../../../devices/cpu/cpu_handle.h"
#include "../../utils.h"
#include "operators.h"

struct SoftmaxCpuDescriptor {
    Device device;
    DT dtype;
    int ndim;
    int *shape;
};

typedef struct SoftmaxCpuDescriptor *SoftmaxCpuDescriptor_t;

infiniopStatus_t cpuCreateSoftmaxDescriptor(infiniopHandle_t handle,
                                            SoftmaxCpuDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t output_desc);


infiniopStatus_t cpuSoftmax(SoftmaxCpuDescriptor_t desc,
                            void const *input,
                            int axis,
                            void *output,
                            void *stream);

infiniopStatus_t cpuDestroySoftmaxDescriptor(SoftmaxCpuDescriptor_t desc);


#endif