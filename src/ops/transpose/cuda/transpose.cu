#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "transpose.cuh"

template<typename Tdata>
__global__ void transpose(
    Tdata *y,
    const Tdata *x,
    uint64_t ndim,
    uint64_t data_size,
    const int64_t *y_strides,
    const int64_t *x_strides,
    const uint64_t *y_shape,
    const uint64_t *x_shape,
    uint64_t offset,
    TransposeMode mode) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < data_size) {
        switch (mode) {
            case TransposeMode::FULL_CONTIGUOUS_COPY:
                y[idx] = x[idx];
                break;
            case TransposeMode::OUTPUT_CONTIGUOUS_COPY:
                y[idx] = x[getOffset(idx, ndim, x_shape, x_strides)];
                break;
            default:// TransposeMode::NON_CONTIGUOUS_COPY:
                y[getOffset(idx, ndim, y_shape, y_strides)] = x[getOffset(idx, ndim, x_shape, x_strides)];
        }
    }
}

template<typename Tdata, unsigned int BLOCK_SIZE>
infiniopStatus_t _transpose_nv_gpu(TransposeCudaDescriptor_t desc, Tdata *y, Tdata const *x, uint64_t data_size, uint64_t pack_size, uint64_t offset, void *stream) {
    if (desc->data_size == 0) {
        return STATUS_SUCCESS;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(BLOCK_SIZE), desc->data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(desc->data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    const auto x_ = reinterpret_cast<Tdata const *>(x);
    const auto y_ = reinterpret_cast<Tdata *>(y);
    const auto x_strides = reinterpret_cast<int64_t const *>(desc->strides_and_shape_d);
    const auto y_strides = reinterpret_cast<int64_t const *>(desc->strides_and_shape_d + desc->stride_size);
    const auto x_shape = reinterpret_cast<uint64_t const *>(desc->strides_and_shape_d + 2 * desc->stride_size);
    const auto y_shape = reinterpret_cast<uint64_t const *>(desc->strides_and_shape_d + 2 * desc->stride_size + desc->shape_size);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < data_size; i += step) {
        transpose<Tdata><<<gridDims, blockDims, 0, cuda_stream>>>(y_, x_, desc->ndim, offset + data_size, y_strides, x_strides, y_shape, x_shape, offset + i, desc->mode);
    }
    return STATUS_SUCCESS;
}

template<typename Tdata, typename TIdata, unsigned int BLOCK_SIZE = 256>
infiniopStatus_t transpose_nv_gpu(TransposeCudaDescriptor_t desc, void *y, void const *x, void *stream, uint64_t pack_size) {
    const auto data_size = desc->data_size / pack_size;
    if (desc->mode == TransposeMode::FULL_CONTIGUOUS_COPY) {
        const auto x_vec = reinterpret_cast<const Tdata *>(x);
        const auto y_vec = reinterpret_cast<Tdata *>(y);
        _transpose_nv_gpu<Tdata, BLOCK_SIZE>(desc, y_vec, x_vec, data_size, pack_size, 0, stream);
    }

    const auto x_ = reinterpret_cast<const TIdata *>(x);
    const auto y_ = reinterpret_cast<TIdata *>(y);
    if (desc->mode == TransposeMode::FULL_CONTIGUOUS_COPY) {
        const auto remainder = desc->data_size % pack_size;
        _transpose_nv_gpu<TIdata, BLOCK_SIZE>(desc, y_, x_, remainder, 1, data_size * pack_size, stream);
    } else {
        _transpose_nv_gpu<TIdata, BLOCK_SIZE>(desc, y_, x_, desc->data_size, 1, 0, stream);
    }
    cudaDeviceSynchronize();
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaTranspose(TransposeCudaDescriptor_t desc,
                               void *y, void const *x,
                               void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return transpose_nv_gpu<float4, half, 256>(desc, y, x, stream, 8);
    }
    if (desc->dtype == F32) {
        return transpose_nv_gpu<float4, float, 256>(desc, y, x, stream, 4);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
