#ifndef __CUDA_GLOBAL_AVG_POOL_H__
#define __CUDA_GLOBAL_AVG_POOL_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>

struct GlobalAvgPoolCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t data_size;
    uint64_t y_data_size;
    uint64_t x_per_NC_data_size;
    unsigned max_block_size;
    uint64_t max_grid_size;
    uint64_t items_per_thread;
    std::shared_ptr<Pool<cudnnHandle_t>> cudnn_handles_t;
    cudnnTensorDescriptor_t const x_desc;
    cudnnTensorDescriptor_t const y_desc;
    cudnnPoolingDescriptor_t const pool_desc;
    const float alpha;
    const float beta;
};

typedef struct GlobalAvgPoolCudaDescriptor *GlobalAvgPoolCudaDescriptor_t;

infiniopStatus_t cudaCreateGlobalAvgPoolDescriptor(CudaHandle_t,
                                                   GlobalAvgPoolCudaDescriptor_t *,
                                                   infiniopTensorDescriptor_t y,
                                                   infiniopTensorDescriptor_t x);

infiniopStatus_t cudaGetGlobalAvgPoolWorkspaceSize(GlobalAvgPoolCudaDescriptor_t desc, uint64_t *size);

infiniopStatus_t cudaGlobalAvgPool(GlobalAvgPoolCudaDescriptor_t desc,
                                   void *workspace, uint64_t workspace_size, void *y, void const *x,
                                   void *stream);

infiniopStatus_t cudaDestroyGlobalAvgPoolDescriptor(GlobalAvgPoolCudaDescriptor_t desc);

#endif
