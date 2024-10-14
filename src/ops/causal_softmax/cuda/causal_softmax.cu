#include "../../../devices/cuda/common_cuda.h"
#include "../../../utils.h"
#include "causal_softmax.cuh"
#include <cub/block/block_reduce.cuh>

struct AttentionCausualMask {
    __forceinline__ __device__ bool
    operator()(int tok_id, int seq_len,
               int pos_id, int total_seq_len) {
        //   tok_id ↓ |<-total_seq_len->|
        //          0 | * * * ... *     |
        //          1 | * * * ... * *   |
        //          2 | * * * ... * * * |
        // seq_len: 3  pos_id->
        return total_seq_len + tok_id >= pos_id + seq_len;
    }
};

template<unsigned int BLOCK_SIZE, class Tdata, class Tmask>
static __device__ void block_padding(
    Tdata *__restrict__ att,
    Tmask mask,
    unsigned int const token_idx,
    unsigned int const seq_len) {
    auto att_idx = threadIdx.x, total_seq_len = blockDim.x;
    auto thread_data = mask(token_idx, seq_len, att_idx, total_seq_len)
                           ? float(att[att_idx])
                           : -__FLT_MAX__;

    using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto block_op = BlockOp(temp_storage);

    __shared__ float max;
    {
        auto acc = block_op.Reduce(thread_data, cub::Max(), total_seq_len);
        if (threadIdx.x == 0) { max = acc; }
    }
    __syncthreads();

    __shared__ float mean;
    {
        auto acc = block_op.Sum(thread_data = expf(thread_data - max), total_seq_len);
        if (threadIdx.x == 0) { mean = fdividef(1, acc); }
    }
    __syncthreads();

    att[att_idx] = Tdata(thread_data * mean);
}

template<unsigned int BLOCK_SIZE, unsigned int ITEMS_PER_THREAD, class Tdata, class Tmask>
static __device__ void block_folding(
    Tdata *__restrict__ att,
    Tmask mask,
    unsigned int const token_idx,
    unsigned int const seq_len,
    unsigned int const total_seq_len) {

    auto local = (total_seq_len + blockDim.x - 1) / blockDim.x;

    auto thread_offset = threadIdx.x * local;
    att += thread_offset;

    float thread_data[ITEMS_PER_THREAD], thread_max = -__FLT_MAX__, thread_sum = 0;
    for (unsigned int i = 0; i < local; ++i) {
        auto att_idx = thread_offset + i;
        thread_data[i] = att_idx < total_seq_len && mask(token_idx, seq_len, att_idx, total_seq_len)
                             ? float(att[i])
                             : -__FLT_MAX__;
        thread_max = cub::Max()(thread_max, thread_data[i]);
    }

    using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto block_op = BlockOp(temp_storage);

    __shared__ float max;
    {
        auto acc = block_op.Reduce(thread_max, cub::Max());
        if (threadIdx.x == 0) { max = acc; }
    }
    __syncthreads();

    __shared__ float mean;
    {
        for (unsigned int i = 0; i < local; ++i) {
            thread_data[i] = expf(thread_data[i] - max);
            thread_sum += thread_data[i];
        }
        auto acc = block_op.Sum(thread_sum);
        if (threadIdx.x == 0) { mean = fdividef(1, acc); }
    }
    __syncthreads();

    for (unsigned int i = 0; i < local; ++i) {
        if (auto att_idx = thread_offset + i; att_idx < total_seq_len) {
            att[i] = Tdata(thread_data[i] * mean);
        }
    }
}

// assert BLOCK_SIZE >= blockDim.x
template<unsigned int BLOCK_SIZE, class Tdata, class Tmask>
static __forceinline__ __device__ void padding(
    Tdata *__restrict__ att,
    Tmask mask,
    int const stride_x,
    int const stride_y,
    int const stride_z) {
    auto offset = blockIdx.x * stride_x + blockIdx.y * stride_y,
         token_idx = blockIdx.y,
         seq_len = gridDim.y;
    block_padding<BLOCK_SIZE>(
        att + offset, mask, token_idx, seq_len);
}

template<unsigned int BLOCK_SIZE, unsigned int ITEMS_PER_THREAD, class Tdata, class Tmask>
static __forceinline__ __device__ void folding(
    Tdata *__restrict__ att,
    Tmask mask,
    unsigned int const total_seq_len,
    int const stride_x,
    int const stride_y,
    int const stride_z) {
    auto offset = blockIdx.x * stride_x + blockIdx.y * stride_y,
         token_idx = blockIdx.y,
         seq_len = gridDim.y;
    block_folding<BLOCK_SIZE, ITEMS_PER_THREAD>(
        att + offset, mask, token_idx, seq_len, total_seq_len);
}

template<unsigned int BLOCK_SIZE, class Tdata>
__global__ void fused_softmax_padding(
    Tdata *__restrict__ att,
    unsigned int const stride_x,
    unsigned int const stride_y,
    unsigned int const stride_z) {

    padding<BLOCK_SIZE>(att, AttentionCausualMask(), stride_x, stride_y, stride_z);
}

template<unsigned int BLOCK_SIZE, unsigned int ITEMS_PER_THREAD, class Tdata>
__global__ void fused_softmax_folding(
    Tdata *__restrict__ att,
    unsigned int const stride_x,
    unsigned int const stride_y,
    unsigned int const stride_z,
    unsigned int const total_seq_len) {
    {
        folding<BLOCK_SIZE, ITEMS_PER_THREAD>(att, AttentionCausualMask(), total_seq_len, stride_x, stride_y, stride_z);
    }
}

template<unsigned int BLOCK_SIZE, class Tdata>
__global__ void fused_softmax_standard(
    Tdata *__restrict__ att_,
    unsigned int const stride_x,
    unsigned int const stride_y,
    unsigned int const stride_z,
    unsigned int const total_seq_len) {
    {
        auto offset = blockIdx.x * stride_x + blockIdx.y * stride_y,
             token_idx = blockIdx.y,
             seq_len = gridDim.y;

        auto att = att_ + offset;
        auto att_idx = threadIdx.x;

        float partial;
        __shared__ float max_;
        __shared__ float sum_;
        using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        auto block_op = BlockOp(temp_storage);

        // Partial max
        partial = -__FLT_MAX__;
        for (unsigned int i = att_idx; i < total_seq_len; i += BLOCK_SIZE) {
            if (i <= total_seq_len - seq_len + token_idx) {
                partial = max(partial, float(att[i]));
            }
        }
        __syncthreads();
        // Block reduce max
        {
            auto acc = block_op.Reduce(partial, cub::Max());
            if (threadIdx.x == 0) { max_ = acc; }
        }
        __syncthreads();

        // Partial sum
        partial = 0.;
        for (unsigned int i = att_idx; i < total_seq_len; i += BLOCK_SIZE) {
            if (i <= total_seq_len - seq_len + token_idx) {
                float e = expf(float(att[i]) - max_);
                partial += e;
            }
        }
        __syncthreads();

        // Block reduce sum
        {
            auto acc = block_op.Reduce(partial, cub::Sum());
            if (threadIdx.x == 0) { sum_ = acc; }
        }
        __syncthreads();

        // Softmax
        for (unsigned int i = att_idx; i < total_seq_len; i += BLOCK_SIZE) {
            if (i <= total_seq_len - seq_len + token_idx) {
                float e = expf(float(att[i]) - max_);
                att[i] = e / sum_;
            } else {
                att[i] = half(0);
            }
        }
    }
}


void causal_softmax_nv_gpu_f16(CausalSoftmaxCudaDescriptor_t desc, void* y, void *stream) {
    unsigned long int total_seq_len = desc->total_seq_len;
    unsigned long int seq_len = desc->seq_len;
    unsigned long int batch_size = desc->batch_size;
    unsigned long int stride_x = desc->stride_b;
    unsigned long int stride_y = desc->stride_i;
    unsigned long int stride_z = desc->stride_j;// covert byte strides to element strides
    unsigned int max_items_per_thread = desc->max_items_per_thread;

    dim3 grid(batch_size, seq_len);
    
    if (max_items_per_thread == 1) {
        fused_softmax_padding<MAX_THREADS_PER_BLOCK>
            <<<grid, total_seq_len, 0, (cudaStream_t) stream>>>((half *) (y), stride_x, stride_y, stride_z);
    } else if (max_items_per_thread <= 16) {
        fused_softmax_folding<MAX_THREADS_PER_BLOCK, 16>
            <<<grid, MAX_THREADS_PER_BLOCK, 0, (cudaStream_t) stream>>>((half *) (y), stride_x, stride_y, stride_z, total_seq_len);
    } else {
        fused_softmax_standard<MAX_THREADS_PER_BLOCK>
            <<<grid, MAX_THREADS_PER_BLOCK, 0, (cudaStream_t) stream>>>((half *) (y), stride_x, stride_y, stride_z, total_seq_len);
    }
}

infiniopStatus_t cudaCausalSoftmax(CausalSoftmaxCudaDescriptor_t desc,
                                   void *workspace,
                                   unsigned long int workspace_size,
                                   void *data,
                                   void *stream){
    if(cudaSetDevice(desc->device_id) != cudaSuccess){
        return STATUS_BAD_DEVICE;
    }
    if (dtype_eq(desc->dtype, F16)){
        causal_softmax_nv_gpu_f16(desc, data, stream);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
