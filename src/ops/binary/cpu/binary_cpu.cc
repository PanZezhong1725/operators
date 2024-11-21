#include "binary_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateBinaryDescriptor(infiniopHandle_t,
                                           BinaryCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t c,
                                           infiniopTensorDescriptor_t a,
                                           infiniopTensorDescriptor_t b,
                                           int mode) {
    uint64_t ndim = c->ndim;
    if (!isValidBroadcastShape(a, b, c)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(c)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (c->dt != F16 && c->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (c->dt != a->dt || c->dt != b->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    bool broadcasted = false;
    if (ndim != a->ndim || ndim != b->ndim) {
        broadcasted = true;
    } else {
        for (uint64_t i = 0; i < ndim; ++i) {
            if (c->shape[i] != a->shape[i] || c->shape[i] != b->shape[i]) {
                broadcasted = true;
                break;
            }
        }
    }
    if (mode < 0 || mode >= BinaryMode::numBinaryMode) {
        return STATUS_BAD_PARAM;
    }
    // bitwise operations are only valid for integral types
    if (c->dt.exponent != 0 && (mode == BinaryMode::BitwiseAnd || mode == BinaryMode::BitwiseOr || mode == BinaryMode::BitwiseXor)) {
        return STATUS_BAD_PARAM;
    }

    uint64_t c_data_size = std::accumulate(c->shape, c->shape + c->ndim, 1ULL, std::multiplies<uint64_t>());

    int64_t *a_strides = nullptr;
    int64_t *b_strides = nullptr;
    int64_t *c_strides = nullptr;
    uint64_t *c_shape = new uint64_t[ndim];
    std::copy(c->shape, c->shape + ndim, c_shape);

    if (broadcasted) {
        // get the adjusted strides for a and b
        a_strides = new int64_t[ndim];
        b_strides = new int64_t[ndim];
        c_strides = new int64_t[ndim];
        for (size_t i = 0; i < ndim; ++i) {
            a_strides[i] = (i < ndim - a->ndim || c->shape[i] != a->shape[i + a->ndim - ndim]) ? 0 : a->strides[i + a->ndim - ndim];
            b_strides[i] = (i < ndim - b->ndim || c->shape[i] != b->shape[i + b->ndim - ndim]) ? 0 : b->strides[i + b->ndim - ndim];
            c_strides[i] = c->strides[i];
        }
    }

    *desc_ptr = new BinaryCpuDescriptor{
        DevCpu,
        c->dt,
        ndim,
        c_data_size,
        c_shape,
        a_strides,
        b_strides,
        c_strides,
        broadcasted,
        mode,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyBinaryDescriptor(BinaryCpuDescriptor_t desc) {
    delete[] desc->a_strides;
    delete[] desc->b_strides;
    delete[] desc->c_strides;
    delete[] desc->c_shape;
    delete desc;
    return STATUS_SUCCESS;
}

// binary elementwise kernel that performs the actual binary computation
template<typename Tdata>
Tdata binary_kernel(const Tdata &a, const Tdata &b, int mode) {
    switch (mode) {
        // Arithmetic operations:
        case BinaryMode::Add:
            return a + b;
        case BinaryMode::Subtract:
            return a - b;
        case BinaryMode::Multiply:
            return a * b;
        case BinaryMode::Divide:
            return a / b;
        case BinaryMode::Pow:
            return std::pow(a, b);
        case BinaryMode::Mod:
            if constexpr (std::is_floating_point_v<Tdata>) {
                return std::fmod(a, b);
            } else {
                return a % b;
            }
        case BinaryMode::Max:
            if constexpr (std::is_floating_point_v<Tdata>) {
                return std::fmax(a, b);
            } else {
                return std::max(a, b);
            }
        case BinaryMode::Min:
            if constexpr (std::is_floating_point_v<Tdata>) {
                return std::fmin(a, b);
            } else {
                return std::min(a, b);
            }
        case BinaryMode::BitwiseAnd:
            if constexpr (std::is_integral_v<Tdata>) {
                return a & b;
            } else {
                ASSERT(false);
            }
        case BinaryMode::BitwiseOr:
            if constexpr (std::is_integral_v<Tdata>) {
                return a | b;
            } else {
                ASSERT(false);
            }
        case BinaryMode::BitwiseXor:
            if constexpr (std::is_integral_v<Tdata>) {
                return a ^ b;
            } else {
                ASSERT(false);
            }
        // Logical operations:
        default:
            fprintf(stderr, "Unrecognized binary operation: how did you get here??\n");
            ASSERT(false);
            return a;
    }
}

template<typename Tdata>
infiniopStatus_t binary_cpu(BinaryCpuDescriptor_t desc, void *c, void const *a, void const *b) {
    auto a_ = reinterpret_cast<Tdata const *>(a);
    auto b_ = reinterpret_cast<Tdata const *>(b);
    auto c_ = reinterpret_cast<Tdata *>(c);

#pragma omp parallel for
    for (uint64_t i = 0; i < desc->c_data_size; ++i) {
        auto a_index = desc->broadcasted ? getDstOffset(i, desc->ndim, desc->c_strides, desc->a_strides) : i;
        auto b_index = desc->broadcasted ? getDstOffset(i, desc->ndim, desc->c_strides, desc->b_strides) : i;
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            c_[i] = f32_to_f16(binary_kernel(f16_to_f32(a_[a_index]), f16_to_f32(b_[b_index]), desc->mode));
        } else {
            c_[i] = binary_kernel(a_[a_index], b_[b_index], desc->mode);
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuBinary(BinaryCpuDescriptor_t desc,
                           void *c, void const *a, void const *b,
                           void *stream) {
    if (desc->dtype == F16) {
        return binary_cpu<uint16_t>(desc, c, a, b);
    }
    if (desc->dtype == F32) {
        return binary_cpu<float>(desc, c, a, b);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
