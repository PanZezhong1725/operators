#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "add.cuh"

template<typename T, int N>
struct vecN {
    T data[N];

    __device__ vecN operator+(const vecN<T, N> &other) const {
        vecN<T, N> result;
        for (int i = 0; i < N; ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    __device__ const T &operator[](int i) const {
        return data[i];
    }
};

__device__ uint64_t getDstIndex(uint64_t flat_index, uint64_t ndim, int64_t const *src_strides, int64_t const *dst_strides) {
    uint64_t res = 0;
    for (uint64_t i = 0; i < ndim; ++i) {
        res += flat_index / src_strides[i] * dst_strides[i];
        flat_index %= src_strides[i];
    }
    return res;
}

template<typename Tdata, typename BTdata>
__global__ void add(
    Tdata *c,
    const Tdata *a,
    const Tdata *b,
    const int64_t *a_strides,
    const int64_t *b_strides,
    const int64_t *c_strides,
    uint64_t data_size,
    uint64_t ndim,
    uint64_t offset,
    bool broadcasted,
    unsigned pack_size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < data_size) {
        if (broadcasted) {
            idx *= pack_size;
            auto a_ = reinterpret_cast<const BTdata *>(a);
            auto b_ = reinterpret_cast<const BTdata *>(b);
            auto c_ = reinterpret_cast<BTdata *>(c);
#pragma unroll
            for (size_t i = 0; i < pack_size; ++i) {
                auto a_idx = getDstIndex(idx + i, ndim, c_strides, a_strides);
                auto b_idx = getDstIndex(idx + i, ndim, c_strides, b_strides);
                c_[idx + i] = a_[a_idx] + b_[b_idx];
            }
            return;
        }
        c[idx] = a[idx] + b[idx];
    }
}

template<typename Tdata, typename BTdata>
void _add_nv_gpu(AddCudaDescriptor_t desc, Tdata *c, Tdata const *a, Tdata const *b, uint64_t data_size, uint64_t pack_size, uint64_t offset, void *stream) {
    if (data_size == 0) {
        return;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(MAX_THREADS_PER_BLOCK), data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < data_size; i += step) {
        add<Tdata, BTdata><<<gridDims, blockDims, 0, cuda_stream>>>(
            c, a, b, desc->a_strides, desc->b_strides, desc->c_strides, offset + data_size, desc->ndim, offset + i, desc->broadcasted, pack_size);
    }
}

template<typename Tdata, typename TIdata>
infiniopStatus_t add_nv_gpu(AddCudaDescriptor_t desc, void *c, void const *a, void const *b, void *stream, uint64_t pack_size) {
    auto data_size = desc->c_data_size / pack_size;
    auto a_vec = reinterpret_cast<const Tdata *>(a);
    auto b_vec = reinterpret_cast<const Tdata *>(b);
    auto c_vec = reinterpret_cast<Tdata *>(c);
    _add_nv_gpu<Tdata, TIdata>(desc, c_vec, a_vec, b_vec, data_size, pack_size, 0, stream);

    auto remainder = desc->c_data_size % pack_size;
    auto a_ = reinterpret_cast<const TIdata *>(a);
    auto b_ = reinterpret_cast<const TIdata *>(b);
    auto c_ = reinterpret_cast<TIdata *>(c);
    _add_nv_gpu<TIdata, TIdata>(desc, c_, a_, b_, remainder, 1, data_size * pack_size, stream);
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaAdd(AddCudaDescriptor_t desc,
                         void *c, void const *a, void const *b,
                         void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return add_nv_gpu<vecN<half2, 4>, half>(desc, c, a, b, stream, 8);
    }
    if (desc->dtype == F32) {
        return add_nv_gpu<vecN<float, 4>, float>(desc, c, a, b, stream, 4);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
