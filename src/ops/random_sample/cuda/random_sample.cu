#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "random_sample.cuh"
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

template<class T, int BLOCK_DIM>
__global__ void softmax(
    T *val_out,
    int topk,
    float temperature, int voc) {
    float sum_s = 0.0f;
    for (int i = threadIdx.x; i < topk; i += BLOCK_DIM) {
        sum_s += __expf(static_cast<float>(val_out[i] - val_out[0]) / temperature);
    }
    __shared__ float sum_inverse_total;

    typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Reduce(sum_s, cub::Sum());
    if (threadIdx.x == 0) {
        sum_inverse_total = __fdividef(1.0F, block_sum);//高精度除法
    }

    __syncthreads();
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < topk) {
        val_out[tid] = static_cast<T>(__expf(static_cast<float>(val_out[tid] - val_out[0]) / temperature) * sum_inverse_total);
    }
}

__global__ void index(int *key_in, int voc) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < voc) {
        key_in[ind] = ind;
    }
}
template<class T>
__global__ void random_sample_kernel(int *result,
                                     T *val_out,
                                     float random_val,
                                     float topp,
                                     int topk,
                                     int *key_out) {
    int end = 0;
    for (end = 0; end < topk; end++) {
        if (val_out[end] >= static_cast<T>(topp)) {
            break;
        }
    }
    if (end < topk - 1) {
        end += 1;
    } else {
        end = topk;
    }

    random_val *= static_cast<float>(val_out[end - 1]);
    for (int i = 0; i < end; i++) {
        if (random_val < static_cast<float>(val_out[i])) {
            result[0] = key_out[i];
            break;
        }
    }
}
template<class T, class I>
void sort_pairs_descending(
    void *workspace, size_t &size_radix_sort,
    T const *val_in, T *val_out,
    I *key_in, I *key_out,
    int voc, cudaStream_t stream) {
    cub::DeviceRadixSort::SortPairsDescending(
        workspace, size_radix_sort,
        val_in, val_out,
        key_in, key_out,
        voc, 0, sizeof(T) * 8, stream);
}
template<class T>
void inclusive_sum(
    void *workspace, size_t &size_scan,
    T *data, int voc,
    cudaStream_t stream) {
    cub::DeviceScan::InclusiveSum(
        workspace, size_scan,
        data, data, voc,
        stream);
}
template<class T, class I>
void random_sample_workspace(void *workspace, size_t &size_radix_sort, size_t &size_scan,
                             int voc, cudaStream_t stream) {


    sort_pairs_descending<T, I>(nullptr, size_radix_sort,
                                nullptr, nullptr,
                                nullptr, nullptr,
                                voc, stream);

    inclusive_sum<T>(
        nullptr, size_scan,
        nullptr, voc,
        stream);
}
void random_sample_nv_gpu_f16(RandomSampleCudaDescriptor_t desc, void *workspace, void *result,
                              void *probs,
                              float random_val,
                              float topp,
                              int topk,
                              float temperature,
                              void *stream) {
    int voc = desc->voc;
    //下面这段代码在排序


    half *val_out;
    cudaMalloc((void **) &val_out, voc * sizeof(half));
    int *key_in, *key_out;
    cudaMalloc((void **) &key_in, voc * sizeof(int));
    cudaMalloc((void **) &key_out, voc * sizeof(int));
    index<<<(voc + 1023) / 1024, 1024, 0, (cudaStream_t) stream>>>(key_in, voc);
    //下面开始计算workspace空间
    size_t size_radix_sort;
    size_t size_scan;
    random_sample_workspace<half, int>(workspace, size_radix_sort, size_scan,
                                       voc, (cudaStream_t) stream);

    cudaMalloc(&workspace, size_radix_sort + size_scan);
    sort_pairs_descending<half, int>(
        workspace, size_radix_sort,
        (half *) probs, val_out,
        key_in, key_out,
        voc, (cudaStream_t) stream);//该函数会把排序结果和对应索引保存在val_out和key_out上
    //排序结束，然后开始做softmax变换

    int BLOCK_DIM = 1024;
    int num_blocks = (voc + BLOCK_DIM - 1) / BLOCK_DIM;
    softmax<half, 1024><<<num_blocks, BLOCK_DIM, 0, (cudaStream_t) stream>>>(val_out, topk,
                                                                             temperature, voc);


    inclusive_sum<half>(
        workspace, size_scan,
        val_out, voc,
        (cudaStream_t) stream);//该函数会实现scan功能不断累加结果
    random_sample_kernel<half><<<1, 1, 0, (cudaStream_t) stream>>>((int *) result,
                                                                   val_out,
                                                                   random_val,
                                                                   topp,
                                                                   topk,
                                                                   key_out);
    cudaFree(val_out);
    cudaFree(key_in);
    cudaFree(key_out);
}

infiniopStatus_t cudaRandomSample(RandomSampleCudaDescriptor_t desc,
                                  void *workspace,
                                  uint64_t workspace_size,
                                  void *result,
                                  void *probs,
                                  float random_val,
                                  float topp,
                                  int topk,
                                  float temperature,
                                  void *stream) {
    if (cudaSetDevice(desc->device_id) != cudaSuccess) {
        return STATUS_BAD_DEVICE;
    }
    if (dtype_eq(desc->dtype, F16)) {
        random_sample_nv_gpu_f16(desc, workspace, result, probs, random_val, topp, topk, temperature, stream);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
