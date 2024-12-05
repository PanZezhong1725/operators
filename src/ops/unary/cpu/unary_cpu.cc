#include "unary_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateUnaryDescriptor(infiniopHandle_t,
                                          UnaryCpuDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x,
                                          int mode) {
    uint64_t ndim = y->ndim;
    if (ndim != x->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!std::equal(y->shape, y->shape + ndim, x->shape)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (mode < 0 || mode >= UnaryMode::numUnaryMode) {
        return STATUS_BAD_PARAM;
    }
    // bitwise operations are only valid for integral types
    if (y->dt.exponent != 0 && mode == UnaryMode::BitwiseNot) {
        return STATUS_BAD_PARAM;
    }

    uint64_t data_size = std::accumulate(y->shape, y->shape + ndim, 1ULL, std::multiplies<uint64_t>());

    *desc_ptr = new UnaryCpuDescriptor{
        DevCpu,
        y->dt,
        ndim,
        data_size,
        mode,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyUnaryDescriptor(UnaryCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

// unary elementwise kernel that performs the actual unary computation
template<typename Tdata>
Tdata unary_kernel(const Tdata &x, int mode) {
    switch (mode) {
        // Arithmetic operations:
        case UnaryMode::Abs:
            return std::abs(x);
        case UnaryMode::Exp:
            return std::exp(x);
        case UnaryMode::Log:
            return std::log(x);
        case UnaryMode::Reciprocal:
            return 1. / x;
        case UnaryMode::Sqrt:
            return std::sqrt(x);
        case UnaryMode::Neg:
            return -x;
        case UnaryMode::Ceil:
            if constexpr (std::is_integral_v<Tdata>) {
                return x;
            } else {
                return std::ceil(x);
            }
        case UnaryMode::Floor:
            if constexpr (std::is_integral_v<Tdata>) {
                return x;
            } else {
                return std::floor(x);
            }
        case UnaryMode::Round:
            if constexpr (std::is_integral_v<Tdata>) {
                return x;
            } else {
                return std::nearbyint(x);
            }
        case UnaryMode::Sin:
            return std::sin(x);
        case UnaryMode::Cos:
            return std::cos(x);
        case UnaryMode::Tan:
            return std::tan(x);
        case UnaryMode::Asin:
            return std::asin(x);
        case UnaryMode::Acos:
            return std::acos(x);
        case UnaryMode::Atan:
            return std::atan(x);
        case UnaryMode::Sinh:
            return std::sinh(x);
        case UnaryMode::Cosh:
            return std::cosh(x);
        case UnaryMode::Tanh:
            return std::tanh(x);
        case UnaryMode::Asinh:
            return std::asinh(x);
        case UnaryMode::Acosh:
            return std::acosh(x);
        case UnaryMode::Atanh:
            return std::atanh(x);
        case UnaryMode::Relu:
            return x < 0 ? 0 : x;
        case UnaryMode::Sigmoid:
            return 1.0 / (1.0 + std::exp(-x));
        case UnaryMode::Sign:
            return x > 0 ? 1 : (x == 0 ? 0 : -1);
        case UnaryMode::Erf:
            return std::erf(x);
        case UnaryMode::BitwiseNot:
            if constexpr (std::is_integral_v<Tdata>) {
                return ~x;
            } else {
                printf("Error: Non-integral input to bitwise operator\n");
                return x;
            }
        // Logical operations:
        default:
            fprintf(stderr, "Error: Unrecognized unary operation\n");
            ASSERT(false);
            return x;
    }
}

template<typename Tdata>
infiniopStatus_t unary_cpu(UnaryCpuDescriptor_t desc, void *y, void const *x) {
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);

#pragma omp parallel for
    for (uint64_t i = 0; i < desc->data_size; ++i) {
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            y_[i] = f32_to_f16(unary_kernel(f16_to_f32(x_[i]), desc->mode));
        } else {
            y_[i] = unary_kernel(x_[i], desc->mode);
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuUnary(UnaryCpuDescriptor_t desc,
                          void *y, void const *x,
                          void *stream) {
    std::fesetround(FE_TONEAREST);
    if (desc->dtype == F16) {
        return unary_cpu<uint16_t>(desc, y, x);
    }
    if (desc->dtype == F32) {
        return unary_cpu<float>(desc, y, x);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}