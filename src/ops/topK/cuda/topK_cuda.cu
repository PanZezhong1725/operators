#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "topK_cuda.cuh"
#include <queue>
#include <vector>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

struct Compare {
    __host__ __device__ bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first > b.first;
    }
};

__host__ __device__ int2 make_pair(int first, int second) {
    int2 p;
    p.x = first;
    p.y = second;
    return p;
}

__host__ __device__ void swap(int2 &a, int2 &b) {
    int temp = a.x;
    a.x = b.x;
    b.x = temp;
    temp = a.y;
    a.y = b.y;
    b.y = temp;
}

template<unsigned int BLOCK_SIZE>
__global__ void topK_kernel(int *indexData, int *probData, int *logitData, uint64_t totalElements, int64_t k) {
    extern __shared__ int2 s_data[];

    int tid = threadIdx.x;
    int currIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (currIdx < totalElements) {
        s_data[tid] = make_pair(indexData[currIdx], currIdx);
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        int mirr = (tid + s - 1) % blockDim.x;
        if (tid < s) {
            if (s_data[tid].x < s_data[mirr].x) {
                swap(s_data[tid], s_data[mirr]);
            }
        }
        __syncthreads();
    }

    for (int s = 1; s <= blockDim.x; s *= 2) {
        int mirr = (tid + s - 1) % blockDim.x;
        if (tid < s) {
            if (s_data[tid].x < s_data[mirr].x) {
                swap(s_data[tid], s_data[mirr]);
            }
        }
        __syncthreads();
    }

    if (tid < k) {
        probData[blockIdx.x * k + tid] = s_data[tid].x;
        logitData[blockIdx.x * k + tid] = s_data[tid].y;
    }
}

void topK_nv_gpu_f16(Tensor indices, Tensor probs, Tensor logits, int64_t k, void *stream) {
    ASSERT(k > 0);

    int *indexData = static_cast<int *>(indices.data);
    int *probData = static_cast<int *>(probs.data);
    int *logitData = static_cast<int *>(logits.data);

    uint64_t ndim = indices.layout->ndim;
    uint64_t *shape = indices.layout->shape;
    int64_t *strides = indices.layout->strides;

    uint64_t totalElements = 1;
    for (uint64_t i = 0; i < ndim; ++i) {
        totalElements *= shape[i];
    }
    ASSERT(k < totalElements);

    size_t bytes = totalElements * sizeof(std::pair<int, int>);
    std::pair<int, int> *d_s_data;
    cudaMalloc(&d_s_data, bytes);

    int batch_size = (totalElements + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    topK_kernel<MAX_THREADS_PER_BLOCK>
        <<<batch_size, MAX_THREADS_PER_BLOCK, 0, (cudaStream_t) stream>>>(indexData, probData, logitData, totalElements, k);

    cudaMemcpy(probData, d_s_data, k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(logitData, d_s_data, k * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_s_data);
}
