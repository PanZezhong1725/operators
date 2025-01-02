#ifndef __CUDA_CONCAT_H__
#define __CUDA_CONCAT_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <vector>
#include <numeric>

struct ConcatCudaDescriptor {
    Device device;
    DT dtype;
    int64_t axis;
    uint64_t num_inputs;
    std::vector<std::vector<uint64_t>> input_shapes;  
    std::vector<uint64_t> output_shape;          
};

typedef struct ConcatCudaDescriptor *ConcatCudaDescriptor_t;

infiniopStatus_t cudaCreateConcatDescriptor(CudaHandle_t handle,
                                            ConcatCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t *x,
                                            uint64_t nums_input,
                                            int64_t axis);

infiniopStatus_t cudaConcat(ConcatCudaDescriptor_t desc,
                            void *y,
                            void const **x,
                            void *stream);

infiniopStatus_t cudaDestroyConcatDescriptor(ConcatCudaDescriptor_t desc);

#endif