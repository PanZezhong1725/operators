#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "rms_norm.cuh"
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

struct Compare {
    __host__ __device__ bool operator()(const pair<int, int>& a, const pair<int, int>& b) {
        return a.first > b.first;
    }
};

template<unsigned int BLOCK_SIZE, class Tdata, class Tmask>
__global__ void topK_kernel(int *indexData, int *probData, int *logitData, uint64_t totalElements, int64_t k) {
    extern __shared__ pair<int, int> s_data[];

    int tid = threadIdx.x;
    int currIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (currIdx < totalElements) {
        s_data[tid] = make_pair(indexData[currIdx], currIdx);
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_data[tid].first < s_data[tid + s].first) {
                swap(s_data[tid], s_data[tid + s]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        maxHeap.push(s_data[0]);
    }
    __syncthreads();

    for (int i = 1; i < k && i < totalElements; i++) {
        int idx = blockIdx.x * blockDim.x + i;
        if (idx < totalElements && indexData[idx] > maxHeap.top().first) {
            maxHeap.pop();
            maxHeap.push(make_pair(indexData[idx], idx));
        }
    }
}

void topK_nv_gpu_f16(Tensor indices, Tensor probs, Tensor logits, int64_t k) {
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

    size_t bytes = totalElements * sizeof(pair<int, int>);
    pair<int, int> *d_s_data;
    cudaMalloc(&d_s_data, bytes);

    std::priority_queue<pair<int, int>, vector<pair<int, int>>, Compare> maxHeap;

    int batch_size = (totalElements + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    topK_kernel<MAX_THREADS_PER_BLOCK>
        <<<batch_size, MAX_THREADS_PER_BLOCK, 0, (cudaStream_t) stream>>>(indexData, probData, logitData, totalElements, k);

    cudaMemcpy(probData, d_s_data, k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(logitData, d_s_data, k * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_s_data);
}
