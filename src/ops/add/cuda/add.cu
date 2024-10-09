#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "add.cuh"

struct half4 {
    __half x, y, z, w;

    __device__ half4 operator+(const half4 &other) const {
        return half4{__hadd(x, other.x), __hadd(y, other.y), __hadd(z, other.z), __hadd(w, other.w)};
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
void add_nv_gpu(AddCudaDescriptor_t desc, Tdata *c, Tdata const *a, Tdata const *b, uint64_t data_size, uint64_t pack_size, uint64_t offset, void *stream) {
    if (data_size == 0) {
        return;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(MAX_THREADS_PER_BLOCK), data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    for (uint64_t i = 0; i < data_size; i += step) {
        add<Tdata, BTdata><<<gridDims, blockDims, 0, cuda_stream>>>(
            c, a, b, desc->a_strides, desc->b_strides, desc->c_strides, offset + data_size, desc->ndim, offset + i, desc->broadcasted, pack_size);
    }
}

void add_nv_gpu_f16(AddCudaDescriptor_t desc, void *c, void const *a, void const *b, void *stream) {
    auto data_size = desc->c_data_size / 4;
    auto a_half4 = reinterpret_cast<const half4 *>(a);
    auto b_half4 = reinterpret_cast<const half4 *>(b);
    auto c_half4 = reinterpret_cast<half4 *>(c);
    add_nv_gpu<half4, half>(desc, c_half4, a_half4, b_half4, data_size, 4, 0, stream);

    auto remainder = desc->c_data_size % 4;
    auto a_half = reinterpret_cast<const half *>(a);
    auto b_half = reinterpret_cast<const half *>(b);
    auto c_half = reinterpret_cast<half *>(c);
    add_nv_gpu<half, half>(desc, c_half, a_half, b_half, remainder, 1, data_size * 4, stream);
}

infiniopStatus_t cudaAdd(AddCudaDescriptor_t desc,
                         void *c, void const *a, void const *b,
                         void *stream) {
    if (!dtype_eq(desc->dtype, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    checkCudaError(cudaSetDevice(desc->device_id));
    add_nv_gpu_f16(desc, c, a, b, stream);
    return STATUS_SUCCESS;
}
