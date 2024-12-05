#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "unary.cuh"
#include <cuda_fp16.h>

/**
 * @brief A templated vector type that supports Unary elementwise operation.
 *
 * @tparam T - The access data type for elements in the vector.
 * @tparam TComp - The computation data type used for arithmetic operations. 
 * @tparam N - The number of elements of type T in the vector for a single access.
 */
template<typename T, typename TComp, size_t N>
struct vecN {
    T data[N];
    constexpr static size_t pack_size = sizeof(T) / sizeof(TComp);

    __device__ __forceinline__ vecN compute(UnaryMode::Mode mode) const {
        vecN<T, TComp, N> result;
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if constexpr (std::is_same_v<T, TComp>) {
                result[i] = unary_op(data[i], mode);
            } else {
                auto &data_ = reinterpret_cast<vecN<TComp, TComp, pack_size> *>(result.data)[i];
                const auto this_data = reinterpret_cast<const vecN<TComp, TComp, pack_size> *>(data)[i];
                data_ = unary_op(this_data, mode);
            }
        }
        return result;
    }

    // Overloaded operators that call the compute method with the appropriate mode.
    __device__ __forceinline__ vecN abs() const { return compute(UnaryMode::Abs); }
    __device__ __forceinline__ vecN exp() const { return compute(UnaryMode::Exp); }
    __device__ __forceinline__ vecN log() const { return compute(UnaryMode::Log); }
    __device__ __forceinline__ vecN reciprocal() const { return compute(UnaryMode::Reciprocal); }
    __device__ __forceinline__ vecN sqrt() const { return compute(UnaryMode::Sqrt); }
    __device__ __forceinline__ vecN operator-() const { return compute(UnaryMode::Neg); }
    __device__ __forceinline__ vecN ceil() const { return compute(UnaryMode::Ceil); }
    __device__ __forceinline__ vecN floor() const { return compute(UnaryMode::Floor); }
    __device__ __forceinline__ vecN round() const { return compute(UnaryMode::Round); }
    __device__ __forceinline__ vecN sin() const { return compute(UnaryMode::Sin); }
    __device__ __forceinline__ vecN cos() const { return compute(UnaryMode::Cos); }
    __device__ __forceinline__ vecN tan() const { return compute(UnaryMode::Tan); }
    __device__ __forceinline__ vecN asin() const { return compute(UnaryMode::Asin); }
    __device__ __forceinline__ vecN acos() const { return compute(UnaryMode::Acos); }
    __device__ __forceinline__ vecN atan() const { return compute(UnaryMode::Atan); }
    __device__ __forceinline__ vecN sinh() const { return compute(UnaryMode::Sinh); }
    __device__ __forceinline__ vecN cosh() const { return compute(UnaryMode::Cosh); }
    __device__ __forceinline__ vecN tanh() const { return compute(UnaryMode::Tanh); }
    __device__ __forceinline__ vecN asinh() const { return compute(UnaryMode::Asinh); }
    __device__ __forceinline__ vecN acosh() const { return compute(UnaryMode::Acosh); }
    __device__ __forceinline__ vecN atanh() const { return compute(UnaryMode::Atanh); }
    __device__ __forceinline__ vecN relu() const { return compute(UnaryMode::Relu); }
    __device__ __forceinline__ vecN sigmoid() const { return compute(UnaryMode::Sigmoid); }
    __device__ __forceinline__ vecN sign() const { return compute(UnaryMode::Sign); }
    __device__ __forceinline__ vecN erf() const { return compute(UnaryMode::Erf); }
    __device__ __forceinline__ vecN operator~() const { return compute(UnaryMode::BitwiseNot); }

    __device__ __forceinline__ const T &operator[](size_t i) const {
        return data[i];
    }

    __device__ __forceinline__ T &operator[](size_t i) {
        return data[i];
    }
};

template<typename Tdata>
__device__ __forceinline__ Tdata unary_abs(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        return __habs2(x);
    } else if constexpr (std::is_same_v<Tdata, half>) {
        return __habs(x);
    } else if constexpr (std::is_floating_point_v<Tdata>) {
        return std::fabs(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::abs(x);
    } else {
        return x.abs();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_exp(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        return h2exp(x);
    } else if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(__expf(__half2float(x)));
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return __expf(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::exp(x);
    } else {
        return x.exp();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_log(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        return h2log(x);
    } else if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(__logf(__half2float(x)));
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return __logf(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::log(x);
    } else {
        return x.log();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_reciprocal(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        return h2rcp(x);
    } else if constexpr (std::is_same_v<Tdata, half>) {
        return hrcp(x);
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return __frcp_rn(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return 1 / x;
    } else {
        return x.reciprocal();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_sqrt(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        return h2sqrt(x);
    } else if constexpr (std::is_same_v<Tdata, half>) {
        return hsqrt(x);
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return __fsqrt_rn(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::sqrt(x);
    } else {
        return x.sqrt();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_neg(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        return __hneg2(x);
    } else if constexpr (std::is_same_v<Tdata, half>) {
        return __hneg(x);
    } else {
        return -x;
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_ceil(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        return h2ceil(x);
    } else if constexpr (std::is_same_v<Tdata, half>) {
        return hceil(x);
    } else if constexpr (std::is_integral_v<Tdata>) {
        return x;
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::ceil(x);
    } else {
        return x.ceil();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_floor(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        return h2floor(x);
    } else if constexpr (std::is_same_v<Tdata, half>) {
        return hfloor(x);
    } else if constexpr (std::is_integral_v<Tdata>) {
        return x;
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::floor(x);
    } else {
        return x.floor();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_round(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        return h2rint(x);
    } else if constexpr (std::is_same_v<Tdata, half>) {
        return hrint(x);
    } else if constexpr (std::is_integral_v<Tdata>) {
        return x;
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::nearbyint(x);
    } else {
        return x.round();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_sin(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        return h2sin(x);
    } else if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(__sinf(__half2float(x)));
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return __sinf(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::sin(x);
    } else {
        return x.sin();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_cos(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        return h2cos(x);
    } else if constexpr (std::is_same_v<Tdata, half>) {
        return hcos(x);
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return __cosf(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::cos(x);
    } else {
        return x.cos();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_tan(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        return h2sin(x) / h2cos(x);
    } else if constexpr (std::is_same_v<Tdata, half>) {
        float tan_f = __tanf(__half2float(x));
        if (std::fabs(tan_f) > TAN_THRESHOLD) {
            return __float2half(tanf(__half2float(x)));
        }
        return __float2half(tan_f);
    } else if constexpr (std::is_same_v<Tdata, float>) {
        float tan_f = __tanf(x);
        if (std::fabs(tan_f) > TAN_THRESHOLD) {
            return tanf(x);
        }
        return tan_f;
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::tan(x);
    } else {
        return x.tan();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_asin(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(asinf(__half2float(x)));
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return asinf(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::asin(x);
    } else if constexpr (!std::is_same_v<Tdata, half2>) {
        return x.asin();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_acos(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(acosf(__half2float(x)));
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return acosf(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::acos(x);
    } else if constexpr (!std::is_same_v<Tdata, half2>) {
        return x.acos();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_atan(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(atanf(__half2float(x)));
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return atanf(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::atan(x);
    } else if constexpr (!std::is_same_v<Tdata, half2>) {
        return x.atan();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_sinh(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(sinhf(__half2float(x)));
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return sinhf(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::sinh(x);
    } else if constexpr (!std::is_same_v<Tdata, half2>) {
        return x.sinh();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_cosh(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(coshf(__half2float(x)));
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return coshf(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::cosh(x);
    } else if constexpr (!std::is_same_v<Tdata, half2>) {
        return x.cosh();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_tanh(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(tanh(__half2float(x)));
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return tanh(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::tanh(x);
    } else if constexpr (!std::is_same_v<Tdata, half2>) {
        return x.tanh();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_asinh(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(asinhf(__half2float(x)));
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return asinhf(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::asinh(x);
    } else if constexpr (!std::is_same_v<Tdata, half2>) {
        return x.asinh();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_acosh(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(acoshf(__half2float(x)));
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return acoshf(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::acosh(x);
    } else if constexpr (!std::is_same_v<Tdata, half2>) {
        return x.acosh();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_atanh(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(atanh(__half2float(x)));
    } else if constexpr (std::is_same_v<Tdata, float>) {
        return atanh(x);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::atanh(x);
    } else if constexpr (!std::is_same_v<Tdata, half2>) {
        return x.atanh();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_relu(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        return __hmul2(x, __hgt2(x, make_half2(0.0f, 0.0f)));
    } else if constexpr (std::is_same_v<Tdata, half>) {
        return __hgt(x, Tdata(0)) ? x : Tdata(0);
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return x <= Tdata(0) ? Tdata(0) : x;
    } else {
        return x.relu();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_sigmoid(const Tdata &x) {
    // 1.0 / (1.0 + exp(-x));
    if constexpr (std::is_same_v<Tdata, half2>) {
        return unary_reciprocal(__hadd2(make_half2(1, 1), unary_exp(unary_neg(x))));
    } else if constexpr (std::is_same_v<Tdata, half>) {
        return unary_reciprocal(__hadd(half(1.f), unary_exp(unary_neg(x))));
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return unary_reciprocal(1 + unary_exp(unary_neg(x)));
    } else {
        return x.sigmoid();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_sign(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half2>) {
        const auto lt_mask = __hlt2(x, make_half2(0.0f, 0.0f));
        return __hadd2(unary_neg(lt_mask), __hsub2(make_half2(1.0f, 1.0f), lt_mask));
    } else if constexpr (std::is_same_v<Tdata, half> || std::is_arithmetic_v<Tdata>) {
        return x > Tdata(0) ? Tdata(1) : (x == Tdata(0) ? Tdata(0) : Tdata(-1));
    } else {
        return x.sign();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_erf(const Tdata &x) {
    if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(erff(__half2float(x)));
    } else if constexpr (std::is_arithmetic_v<Tdata>) {
        return std::erf(x);
    } else if constexpr (!std::is_same_v<Tdata, half2>) {
        return x.erf();
    }
}

template<typename Tdata>
__device__ __forceinline__ Tdata unary_op(const Tdata &x, int mode) {
    switch (mode) {
        // Arithmetic operations:
        case UnaryMode::Abs:
            return unary_abs(x);
        case UnaryMode::Exp:
            return unary_exp(x);
        case UnaryMode::Log:
            return unary_log(x);
        case UnaryMode::Reciprocal:
            return unary_reciprocal(x);
        case UnaryMode::Sqrt:
            return unary_sqrt(x);
        case UnaryMode::Neg:
            return unary_neg(x);
        case UnaryMode::Ceil:
            return unary_ceil(x);
        case UnaryMode::Floor:
            return unary_floor(x);
        case UnaryMode::Round:
            return unary_round(x);
        case UnaryMode::Sin:
            return unary_sin(x);
        case UnaryMode::Cos:
            return unary_cos(x);
        case UnaryMode::Tan:
            return unary_tan(x);
        case UnaryMode::Asin:
            return unary_asin(x);
        case UnaryMode::Acos:
            return unary_acos(x);
        case UnaryMode::Atan:
            return unary_atan(x);
        case UnaryMode::Sinh:
            return unary_sinh(x);
        case UnaryMode::Cosh:
            return unary_cosh(x);
        case UnaryMode::Tanh:
            return unary_tanh(x);
        case UnaryMode::Asinh:
            return unary_asinh(x);
        case UnaryMode::Acosh:
            return unary_acosh(x);
        case UnaryMode::Atanh:
            return unary_atanh(x);
        case UnaryMode::Relu:
            return unary_relu(x);
        case UnaryMode::Sigmoid:
            return unary_sigmoid(x);
        case UnaryMode::Sign:
            return unary_sign(x);
        case UnaryMode::Erf:
            return unary_erf(x);
        case UnaryMode::BitwiseNot:
            if constexpr (std::is_integral_v<Tdata>) {
                return ~x;
            } else {
                printf("Error: Non-integral input to bitwise operator\n");
                return x;
            }

        // Logical operations:
        // TODO. Currently Not supported yet
        default:
            printf("Error: Unrecognized Unary operation\n");
            return x;
    }
}

template<typename Tdata>
__global__ void unary(
    Tdata *y,
    const Tdata *x,
    uint64_t data_size,
    uint64_t offset,
    const int mode) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < data_size) {
        y[idx] = unary_op(x[idx], mode);
    }
}

template<typename Tdata, unsigned int BLOCK_SIZE>
void _unary_nv_gpu(UnaryCudaDescriptor_t desc, Tdata *y, Tdata const *x, uint64_t data_size, uint64_t pack_size, uint64_t offset, void *stream) {
    if (data_size == 0) {
        return;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(BLOCK_SIZE), data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < data_size; i += step) {
        unary<Tdata><<<gridDims, blockDims, 0, cuda_stream>>>(y, x, offset + data_size, offset + i, desc->mode);
    }
}

template<typename Tdata, typename TIdata, unsigned int BLOCK_SIZE = 256>
infiniopStatus_t unary_nv_gpu(UnaryCudaDescriptor_t desc, void *y, void const *x, void *stream, uint64_t pack_size) {
    const auto data_size = desc->data_size / pack_size;
    const auto x_vec = reinterpret_cast<const Tdata *>(x);
    const auto y_vec = reinterpret_cast<Tdata *>(y);
    _unary_nv_gpu<Tdata, BLOCK_SIZE>(desc, y_vec, x_vec, data_size, pack_size, 0, stream);

    const auto remainder = desc->data_size % pack_size;
    if (remainder > 0) {
        const auto x_ = reinterpret_cast<const TIdata *>(x);
        const auto y_ = reinterpret_cast<TIdata *>(y);
        _unary_nv_gpu<TIdata, BLOCK_SIZE>(desc, y_, x_, remainder, 1, data_size * pack_size, stream);
    }
    cudaDeviceSynchronize();
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaUnary(UnaryCudaDescriptor_t desc,
                           void *y, void const *x,
                           void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    std::fesetround(FE_TONEAREST);

    if (desc->dtype == F16) {
        switch (desc->mode) {
            case UnaryMode::Abs:
            case UnaryMode::Exp:
            case UnaryMode::Reciprocal:
                return unary_nv_gpu<vecN<float4, half2, 1>, half, 1024>(desc, y, x, stream, 8);
            case UnaryMode::Log:
                return unary_nv_gpu<vecN<float4, half2, 1>, half, 512>(desc, y, x, stream, 8);
            case UnaryMode::Sqrt:
            case UnaryMode::Neg:
            case UnaryMode::Ceil:
            case UnaryMode::Floor:
            case UnaryMode::Round:
            case UnaryMode::Sigmoid:
            case UnaryMode::Sign:
                return unary_nv_gpu<vecN<float2, half2, 2>, half, 64>(desc, y, x, stream, 8);
            case UnaryMode::Sin:
            case UnaryMode::Cos:
            case UnaryMode::Tan:
            case UnaryMode::Asin:
            case UnaryMode::Acos:
            case UnaryMode::Atan:
            case UnaryMode::Sinh:
            case UnaryMode::Cosh:
            case UnaryMode::Tanh:
            case UnaryMode::Asinh:
            case UnaryMode::Acosh:
            case UnaryMode::Atanh:
            case UnaryMode::Erf:
                return unary_nv_gpu<vecN<float2, half, 2>, half, 64>(desc, y, x, stream, 8);
            case UnaryMode::Relu:
                return unary_nv_gpu<vecN<float2, half2, 2>, half, 128>(desc, y, x, stream, 8);
            default:
                return unary_nv_gpu<vecN<float4, half, 1>, half, 64>(desc, y, x, stream, 8);
        }
    }
    if (desc->dtype == F32) {
        switch (desc->mode) {
            case UnaryMode::Abs:
            case UnaryMode::Log:
            case UnaryMode::Reciprocal:
                return unary_nv_gpu<vecN<float2, float, 2>, float, 1024>(desc, y, x, stream, 4);
            case UnaryMode::Exp:
                return unary_nv_gpu<vecN<float4, float, 1>, float, 1024>(desc, y, x, stream, 4);
            case UnaryMode::Sqrt:
            case UnaryMode::Neg:
            case UnaryMode::Ceil:
            case UnaryMode::Floor:
            case UnaryMode::Round:
            case UnaryMode::Sin:
            case UnaryMode::Cos:
            case UnaryMode::Tan:
            case UnaryMode::Asin:
            case UnaryMode::Acos:
            case UnaryMode::Atan:
            case UnaryMode::Sinh:
            case UnaryMode::Cosh:
            case UnaryMode::Tanh:
            case UnaryMode::Asinh:
            case UnaryMode::Acosh:
            case UnaryMode::Atanh:
            case UnaryMode::Sigmoid:
            case UnaryMode::Sign:
            case UnaryMode::Erf:
                return unary_nv_gpu<vecN<float2, float, 2>, float, 64>(desc, y, x, stream, 4);
            case UnaryMode::Relu:
                return unary_nv_gpu<vecN<float4, float, 1>, float, 1024>(desc, y, x, stream, 4);
            default:
                return unary_nv_gpu<vecN<float2, float, 2>, float, 256>(desc, y, x, stream, 4);
        }
    }
    return STATUS_BAD_TENSOR_DTYPE;
}