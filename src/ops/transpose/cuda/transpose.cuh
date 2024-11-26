#ifndef __CUDA_TRANSPOSE_H__
#define __CUDA_TRANSPOSE_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "../transpose_common.h"
#include <cuda_fp16.h>
#include <numeric>

struct TransposeCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t data_size;
    uint64_t max_grid_size;
    char const *strides_and_shape_d;
    size_t const shape_size;
    size_t const stride_size;
    TransposeMode mode;
};

typedef struct TransposeCudaDescriptor *TransposeCudaDescriptor_t;

infiniopStatus_t cudaCreateTransposeDescriptor(CudaHandle_t,
                                               TransposeCudaDescriptor_t *,
                                               infiniopTensorDescriptor_t y,
                                               infiniopTensorDescriptor_t x,
                                               uint64_t const *perm,
                                               uint64_t n);

infiniopStatus_t cudaTranspose(TransposeCudaDescriptor_t desc,
                               void *y, void const *x,
                               void *stream);

infiniopStatus_t cudaDestroyTransposeDescriptor(TransposeCudaDescriptor_t desc);

#endif
