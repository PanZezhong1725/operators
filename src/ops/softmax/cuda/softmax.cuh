#ifndef __CUDA_SOFTMAX_H__
#define __CUDA_SOFTMAX_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "../../utils.h"
#include "operators.h"

struct SoftmaxCudaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    int ndim;
    int *shape;
};

typedef struct SoftmaxCudaDescriptor *SoftmaxCudaDescriptor_t;

infiniopStatus_t cudaCreateSoftmaxDescriptor(CudaHandle_t handle,
                                             SoftmaxCudaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t output_desc);


infiniopStatus_t cudaSoftmax(SoftmaxCudaDescriptor_t desc,
                             void const *input,
                             int axis,
                             void *output,
                             void *stream);

infiniopStatus_t cudaDestroySoftmaxDescriptor(SoftmaxCudaDescriptor_t desc);


#endif