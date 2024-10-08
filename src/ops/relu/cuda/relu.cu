#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "relu.cuh"

namespace infini {
    struct half2 {
        __half x, y;

        // constructor that initializes both components with the same value
        __device__ half2(__half value) : x(value), y(value) {}

        // constructor that initializes with two different values
        __device__ half2(__half value_x, __half value_y) : x(value_x), y(value_y) {}

        // assignment with ReLU logic
        __device__ half2 &operator=(const half2 &other) {
            x = __hgt(other.x, __half(0.0f)) ? other.x : __half(0.0f);
            y = __hgt(other.y, __half(0.0f)) ? other.y : __half(0.0f);
            return *this;
        }
    };
}// namespace infini


template<typename Tdata>
__global__ void relu(
    Tdata *y,
    const Tdata *x,
    uint64_t data_size,
    uint64_t offset) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < data_size) {
        if constexpr (std::is_same<Tdata, infini::half2>::value) {
            y[idx] = x[idx];
        } else {
            y[idx] = x[idx] < Tdata(0) ? Tdata(0) : x[idx];
        }
    }
}

template<typename Tdata>
void relu_nv_gpu(ReluCudaDescriptor_t desc, Tdata *y, Tdata const *x, uint64_t data_size, uint64_t offset, void *stream) {
    if (data_size == 0) {
        return;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(MAX_THREADS_PER_BLOCK), data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    for (uint64_t i = 0; i < data_size; i += step) {
        relu<Tdata><<<gridDims, blockDims, 0, cuda_stream>>>(y, x, offset + data_size, offset + i);
    }
}

void relu_nv_gpu_f16(ReluCudaDescriptor_t desc, void *y, void const *x, void *stream) {
    auto data_size = desc->data_size / 2;
    auto x_half2 = reinterpret_cast<const infini::half2 *>(x);
    auto y_half2 = reinterpret_cast<infini::half2 *>(y);
    relu_nv_gpu(desc, y_half2, x_half2, data_size, 0, stream);

    auto remainder = desc->data_size % 2;
    auto x_half = reinterpret_cast<const half *>(x);
    auto y_half = reinterpret_cast<half *>(y);
    relu_nv_gpu(desc, y_half, x_half, remainder, data_size * 2, stream);
}

infiniopStatus_t cudaRelu(ReluCudaDescriptor_t desc,
                          void *y, void const *x,
                          void *stream) {
    if (desc->dtype != F16) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    checkCudaError(cudaSetDevice(desc->device_id));
    relu_nv_gpu_f16(desc, y, x, stream);
    return STATUS_SUCCESS;
}
