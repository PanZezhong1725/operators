#include "../../utils.h"
#include "random_sample.cuh"
#include <cuda_fp16.h>



template<class Tdata>
static __global__ void random_sample(
    Tdata *__restrict__ source,
    unsigned int const *__restrict__ indices,
    unsigned int *__restrict__ index_,
    float random, float topp, int topk, int voc) {
    Tdata p = static_cast<Tdata>(random * min(topp * static_cast<float>(source[voc - 1]), static_cast<float>(source[topk - 1])));
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < voc){
        if(source[i] >= p){
            index_[0] = indices[i];
        }
    }
}

constexpr static int BLOCK_SIZE = 1024;

void random_sample_nv_gpu_f16(Tensor source, Tensor indices, Tensor index, float random, float topp, int topk, void *stream) {
    ASSERT_EQ(source.layout->ndim, 1);
    ASSERT_EQ(indices.layout->ndim, 1);
    auto voc = source.layout->shape[0];
    auto source_ = reinterpret_cast<half const *>(source.data);
    auto indices_ = reinterpret_cast<int const *>(indeces.data);
    auto index_ = reinterpret_cast<int const *>(index.data);

    
    int num_blocks = (voc + BLOCK_SIZE - 1) / BLOCK_SIZE;

    

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    random_sample<<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
        source_, indices_, index_, random, topp, topk);
}
