#ifndef __CUDA_UNARY_H__
#define __CUDA_UNARY_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "../unary.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct UnaryCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t data_size;
    uint64_t max_grid_size;
    int mode;
};

typedef struct UnaryCudaDescriptor *UnaryCudaDescriptor_t;

infiniopStatus_t cudaCreateUnaryDescriptor(CudaHandle_t,
                                           UnaryCudaDescriptor_t *,
                                           infiniopTensorDescriptor_t y,
                                           infiniopTensorDescriptor_t x,
                                           int mode);

infiniopStatus_t cudaUnary(UnaryCudaDescriptor_t desc,
                           void *c, void const *x,
                           void *stream);

infiniopStatus_t cudaDestroyUnaryDescriptor(UnaryCudaDescriptor_t desc);

#endif