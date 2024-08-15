#include "../../utils.h"
#include "random_sample.cuh"
#include <cuda_fp16.h>



template<class Tdata>
static __global__ void random_sample(
    Tdata *__restrict__ source,
    unsigned int *__restrict__ indices,
    float topp, int topk, float temperature) {
    int ind = threadIdx.x;
}

constexpr static int BLOCK_SIZE = 1024;

void random_sample_nv_gpu_f16(Tensor source, Tensor indices, float topp, int topk, float temperature, void *stream) {
    ASSERT_EQ(source.layout->ndim, 1);
    ASSERT_EQ(indices.layout->ndim, 1);
    auto voc = source.layout->shape[0];
    auto logits_ = reinterpret_cast<half *>(source.data);
    auto index_ = reinterpret_cast<int *>(indeces.data);
    

    
    
}
