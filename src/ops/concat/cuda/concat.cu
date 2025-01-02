#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "concat.cuh"

// Kernel function to perform concatenation on GPU
template <typename T>
__global__ void concatKernel(const T* x, T* y,
                             size_t inSize, 
                             size_t localBlockOffset,
                             size_t innerOffset, 
                             size_t blockOffset) {
    size_t iOffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (iOffset < inSize) {
        size_t oOffset = (iOffset % localBlockOffset) + innerOffset +
                         (iOffset / localBlockOffset) * blockOffset;
        y[oOffset] = x[iOffset];
    }
}

template <typename T>
infiniopStatus_t concatCompute(ConcatCudaDescriptor_t& desc,
                               T* y,
                               void const** x,
                               cudaStream_t stream) {
    int64_t axis = desc->axis;
    uint64_t num_inputs = desc->num_inputs;
    const std::vector<std::vector<uint64_t>>& input_shapes = desc->input_shapes;
    const std::vector<uint64_t>& output_shape = desc->output_shape;

    size_t blockOffsetInner = 1;
    for (size_t i = output_shape.size() - 1; i > axis; --i) {
        blockOffsetInner *= output_shape[i];
    }
    size_t blockOffset = output_shape[axis] * blockOffsetInner;

#pragma unroll
    for (size_t i = 0; i < num_inputs; ++i) {
        const std::vector<uint64_t>& input_shape = input_shapes[i];

        size_t dimOffset = 0;
        for (size_t j = 0; j < i; ++j) {
            dimOffset += input_shapes[j][axis];
        }

        size_t localBlockOffset = 1;
        for (size_t j = input_shape.size() - 1; j >= axis && j != static_cast<size_t>(-1); --j) {
            localBlockOffset *= input_shape[j];
        }

        size_t innerOffset = blockOffsetInner * dimOffset;
        size_t inSize = 1;
        for (auto dim : input_shape) {
            inSize *= dim;
        }

        T* input_data = static_cast<T*>(const_cast<void*>(x[i]));

        // Launch CUDA kernel
        int threads = 256;
        int blocks = (inSize + threads - 1) / threads;
        concatKernel<<<blocks, threads, 0, stream>>>(input_data, y, inSize, localBlockOffset, innerOffset, blockOffset);

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return STATUS_EXECUTION_FAILED;
        }
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaConcat(ConcatCudaDescriptor_t desc,
                            void* y,
                            void const** x,
                            void* stream) {
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream);

    if (desc->dtype == F16) {
        return concatCompute<uint16_t>(desc, reinterpret_cast<uint16_t*>(y), x, cudaStream);
    }
    if (desc->dtype == F32) {
        return concatCompute<float>(desc, reinterpret_cast<float*>(y), x, cudaStream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}