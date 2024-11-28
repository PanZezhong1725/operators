#ifndef __CUDA_BINARY_H__
#define __CUDA_BINARY_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "../binary.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct BinaryCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t c_data_size;
    uint64_t max_grid_size;
    int64_t const *strides_d;
    bool broadcasted;
    int mode;
};

typedef struct BinaryCudaDescriptor *BinaryCudaDescriptor_t;

infiniopStatus_t cudaCreateBinaryDescriptor(CudaHandle_t,
                                            BinaryCudaDescriptor_t *,
                                            infiniopTensorDescriptor_t c,
                                            infiniopTensorDescriptor_t a,
                                            infiniopTensorDescriptor_t b,
                                            int mode);

infiniopStatus_t cudaBinary(BinaryCudaDescriptor_t desc,
                            void *c, void const *a, void const *b,
                            void *stream);

infiniopStatus_t cudaDestroyBinaryDescriptor(BinaryCudaDescriptor_t desc);

#endif
