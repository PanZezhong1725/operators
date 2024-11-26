#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "binary.cuh"
#include <cuda_fp16.h>

/**
 * @brief A templated vector type that supports binary elementwise operation.
 *
 * @tparam T - The access data type for elements in the vector.
 * @tparam TComp - The computation data type used for arithmetic operations. 
 * @tparam N - The number of elements of type T in the vector for a single access.
 */
template<typename T, typename TComp, size_t N>
struct vecN {
    T data[N];
    constexpr static size_t pack_size = sizeof(T) / sizeof(TComp);

    __device__ __forceinline__ vecN compute(const vecN<T, TComp, N> &other, BinaryMode::Mode mode) const {
        vecN<T, TComp, N> result;
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if constexpr (std::is_same_v<T, TComp>) {
                result[i] = binary_op(data[i], other[i], mode);
            } else {
                auto &data_ = reinterpret_cast<vecN<TComp, TComp, pack_size> *>(result.data)[i];
                const auto this_data = reinterpret_cast<const vecN<TComp, TComp, pack_size> *>(data)[i];
                const auto other_data = reinterpret_cast<const vecN<TComp, TComp, pack_size> *>(other.data)[i];
                data_ = binary_op(this_data, other_data, mode);
            }
        }
        return result;
    }

    // Overloaded operators that call the compute method with the appropriate mode.
    __device__ __forceinline__ vecN operator+(const vecN<T, TComp, N> &other) const {
        return compute(other, BinaryMode::Add);
    }

    __device__ __forceinline__ vecN operator-(const vecN<T, TComp, N> &other) const {
        return compute(other, BinaryMode::Subtract);
    }

    __device__ __forceinline__ vecN operator*(const vecN<T, TComp, N> &other) const {
        return compute(other, BinaryMode::Multiply);
    }

    __device__ __forceinline__ vecN operator/(const vecN<T, TComp, N> &other) const {
        return compute(other, BinaryMode::Divide);
    }

    __device__ __forceinline__ vecN operator&(const vecN<T, TComp, N> &other) const {
        return compute(other, BinaryMode::BitwiseAnd);
    }

    __device__ __forceinline__ vecN operator|(const vecN<T, TComp, N> &other) const {
        return compute(other, BinaryMode::BitwiseOr);
    }

    __device__ __forceinline__ vecN operator^(const vecN<T, TComp, N> &other) const {
        return compute(other, BinaryMode::BitwiseXor);
    }

    __device__ __forceinline__ vecN pow(const vecN<T, TComp, N> &other) const {
        return compute(other, BinaryMode::Pow);
    }

    __device__ __forceinline__ vecN mod(const vecN<T, TComp, N> &other) const {
        return compute(other, BinaryMode::Mod);
    }

    __device__ __forceinline__ vecN max(const vecN<T, TComp, N> &other) const {
        return compute(other, BinaryMode::Max);
    }

    __device__ __forceinline__ vecN min(const vecN<T, TComp, N> &other) const {
        return compute(other, BinaryMode::Min);
    }

    __device__ __forceinline__ const T &operator[](size_t i) const {
        return data[i];
    }

    __device__ __forceinline__ T &operator[](size_t i) {
        return data[i];
    }
};


template<typename Tdata>
__device__ __forceinline__ Tdata binary_op(const Tdata &a, const Tdata &b, int mode) {
    switch (mode) {
        // Arithmetic operations:
        case BinaryMode::Add:
            if constexpr (std::is_same_v<Tdata, half2>) {
                return __hadd2_rn(a, b);
            } else if constexpr (std::is_same_v<Tdata, float>) {
                return __fadd_rn(a, b);
            } else {
                return a + b;
            }
        case BinaryMode::Subtract:
            if constexpr (std::is_same_v<Tdata, half2>) {
                return __hsub2(a, b);
            } else if constexpr (std::is_same_v<Tdata, float>) {
                return __fsub_rn(a, b);
            } else {
                return a - b;
            }
        case BinaryMode::Multiply:
            if constexpr (std::is_same_v<Tdata, half2>) {
                return __hmul2(a, b);
            } else if constexpr (std::is_same_v<Tdata, float>) {
                return __fmul_rd(a, b);
            } else {
                return a * b;
            }
        case BinaryMode::Divide:
            return a / b;
        case BinaryMode::Pow:
            if constexpr (std::is_arithmetic_v<Tdata>) {
                return std::pow(a, b);
            } else if constexpr (std::is_same_v<Tdata, half>) {
                return __float2half(std::pow(__half2float(a), __half2float(b)));
            } else if constexpr (!std::is_same_v<Tdata, half2>) {
                return a.pow(b);
            }
            return a;
        case BinaryMode::Mod:
            if constexpr (std::is_floating_point_v<Tdata>) {
                return std::fmod(a, b);
            } else if constexpr (std::is_arithmetic_v<Tdata>) {
                return a % b;
            } else if constexpr (std::is_same_v<Tdata, half>) {
                return __float2half(std::fmod(__half2float(a), __half2float(b)));
            } else if constexpr (!std::is_same_v<Tdata, half2>) {
                return a.mod(b);
            }
            return a;
        case BinaryMode::Max:
            if constexpr (std::is_fundamental_v<Tdata>) {
                if constexpr (std::is_floating_point_v<Tdata>) {
                    return std::fmax(a, b);
                } else {
                    return std::max(a, b);
                }
            } else if constexpr (std::is_same_v<Tdata, half>) {
                return __half2float(a) > __half2float(b) ? a : b;
            } else if constexpr (!std::is_same_v<Tdata, half2>) {
                return a.max(b);
            }
        case BinaryMode::Min:
            if constexpr (std::is_fundamental_v<Tdata>) {
                if constexpr (std::is_floating_point_v<Tdata>) {
                    return std::fmin(a, b);
                } else {
                    return std::min(a, b);
                }
            } else if constexpr (std::is_same_v<Tdata, half>) {
                return __half2float(a) < __half2float(b) ? a : b;
            } else if constexpr (!std::is_same_v<Tdata, half2>) {
                return a.min(b);
            }
        case BinaryMode::BitwiseAnd:
            if constexpr (std::is_integral_v<Tdata>) {
                return a & b;
            } else {
                printf("Error: Non-integral input to bitwise operatior\n");
                return a;
            }
        case BinaryMode::BitwiseOr:
            if constexpr (std::is_integral_v<Tdata>) {
                return a | b;
            } else {
                printf("Error: Non-integral input to bitwise operatior\n");
                return a;
            }
        case BinaryMode::BitwiseXor:
            if constexpr (std::is_integral_v<Tdata>) {
                return a ^ b;
            } else {
                printf("Error: Non-integral input to bitwise operatior\n");
                return a;
            }
        // Logical operations:
        // TODO. Currently Not supported yet
        default:
            printf("Unrecognized binary operation: how did you get here??\n");
            return a;
    }
}

template<typename Tdata, typename BTdata>
__global__ void binary(
    Tdata *c,
    const Tdata *a,
    const Tdata *b,
    const int64_t *__restrict__ a_strides,
    const int64_t *__restrict__ b_strides,
    const int64_t *__restrict__ c_strides,
    uint64_t data_size,
    uint64_t ndim,
    uint64_t offset,
    bool broadcasted,
    unsigned pack_size,
    const int mode) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < data_size) {
        if (broadcasted) {
            idx *= pack_size;
            auto a_ = reinterpret_cast<const BTdata *>(a);
            auto b_ = reinterpret_cast<const BTdata *>(b);
            auto c_ = reinterpret_cast<BTdata *>(c);
#pragma unroll
            for (size_t i = 0; i < pack_size; ++i) {
                auto a_idx = getDstOffset(idx + i, ndim, c_strides, a_strides);
                auto b_idx = getDstOffset(idx + i, ndim, c_strides, b_strides);
                c_[idx + i] = binary_op(a_[a_idx], b_[b_idx], mode);
            }
            return;
        }
        Tdata a_data = a[idx];
        Tdata b_data = b[idx];
        c[idx] = binary_op(a_data, b_data, mode);
    }
}

template<typename Tdata, typename BTdata, unsigned int BLOCK_SIZE>
void _binary_nv_gpu(BinaryCudaDescriptor_t desc, Tdata *c, Tdata const *a, Tdata const *b, uint64_t data_size, uint64_t pack_size, uint64_t offset, void *stream) {
    if (data_size == 0) {
        return;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(BLOCK_SIZE), data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    const int64_t *a_strides = nullptr;
    const int64_t *b_strides = nullptr;
    const int64_t *c_strides = nullptr;
    if (desc->broadcasted) {
        a_strides = desc->strides_d;
        b_strides = desc->strides_d + desc->ndim;
        c_strides = desc->strides_d + 2 * desc->ndim;
    }
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < data_size; i += step) {
        binary<Tdata, BTdata><<<gridDims, blockDims, 0, cuda_stream>>>(
            c, a, b, a_strides, b_strides, c_strides, offset + data_size, desc->ndim, offset + i, desc->broadcasted, pack_size, desc->mode);
    }
}

template<typename Tdata, typename TIdata, unsigned int BLOCK_SIZE = 256>
infiniopStatus_t binary_nv_gpu(BinaryCudaDescriptor_t desc, void *c, void const *a, void const *b, void *stream, uint64_t pack_size) {
    const auto data_size = desc->c_data_size / pack_size;
    const auto a_vec = reinterpret_cast<const Tdata *>(a);
    const auto b_vec = reinterpret_cast<const Tdata *>(b);
    const auto c_vec = reinterpret_cast<Tdata *>(c);
    _binary_nv_gpu<Tdata, TIdata, BLOCK_SIZE>(desc, c_vec, a_vec, b_vec, data_size, pack_size, 0, stream);

    const auto remainder = desc->c_data_size % pack_size;
    if (remainder > 0) {
        const auto a_ = reinterpret_cast<const TIdata *>(a);
        const auto b_ = reinterpret_cast<const TIdata *>(b);
        const auto c_ = reinterpret_cast<TIdata *>(c);
        _binary_nv_gpu<TIdata, TIdata, BLOCK_SIZE>(desc, c_, a_, b_, remainder, 1, data_size * pack_size, stream);
    }
    cudaDeviceSynchronize();
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaBinary(BinaryCudaDescriptor_t desc,
                            void *c, void const *a, void const *b,
                            void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        switch (desc->mode) {
            case BinaryMode::Add:
            case BinaryMode::Subtract:
            case BinaryMode::Multiply:
            case BinaryMode::Divide:
                return binary_nv_gpu<vecN<float2, half2, 2>, half, 256>(desc, c, a, b, stream, 8);
            case BinaryMode::Pow:
                return binary_nv_gpu<vecN<float4, half, 1>, half, 128>(desc, c, a, b, stream, 8);
            default:
                return binary_nv_gpu<vecN<float4, half, 1>, half, 64>(desc, c, a, b, stream, 8);
        }
    }
    if (desc->dtype == F32) {
        switch (desc->mode) {
            case BinaryMode::Add:
            case BinaryMode::Subtract:
            case BinaryMode::Multiply:
            case BinaryMode::Divide:
                return binary_nv_gpu<vecN<float2, float, 2>, float, 256>(desc, c, a, b, stream, 4);
            case BinaryMode::Pow:
                return binary_nv_gpu<vecN<float4, float, 2>, float, 256>(desc, c, a, b, stream, 8);
            case BinaryMode::Mod:
                return binary_nv_gpu<vecN<float4, float, 1>, float, 128>(desc, c, a, b, stream, 4);
            case BinaryMode::Max:
            case BinaryMode::Min:
                return binary_nv_gpu<vecN<float2, float, 2>, float, 64>(desc, c, a, b, stream, 4);
            default:
                return binary_nv_gpu<vecN<float2, float, 2>, float, 256>(desc, c, a, b, stream, 4);
        }
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
