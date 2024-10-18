#include "add_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

inline void incrementOne(uint64_t *indices, uint64_t const *shape, uint64_t ndim) {
    for (int64_t i = ndim - 1; i >= 0; --i) {
        if (++indices[i] != shape[i]) {
            return;
        }
        indices[i] = 0;
    }
}

inline uint64_t compactToFlat(uint64_t const *indices, uint64_t const *strides, uint64_t ndim) {
    return std::inner_product(indices, indices + ndim, strides, uint64_t(0));
}

infiniopStatus_t cpuCreateAddDescriptor(infiniopHandle_t,
                                        AddCpuDescriptor_t *desc_ptr,
                                        infiniopTensorDescriptor_t c,
                                        infiniopTensorDescriptor_t a,
                                        infiniopTensorDescriptor_t b) {
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

    uint64_t c_data_size = std::accumulate(c->shape, c->shape + c->ndim, 1ULL, std::multiplies<uint64_t>());

    // get the adjusted strides for a and b
    uint64_t *a_strides = new uint64_t[ndim];
    uint64_t *b_strides = new uint64_t[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        a_strides[i] = (i < ndim - a->ndim || c->shape[i] != a->shape[i + a->ndim - ndim]) ? 0 : a->strides[i + a->ndim - ndim];
        b_strides[i] = (i < ndim - b->ndim || c->shape[i] != b->shape[i + b->ndim - ndim]) ? 0 : b->strides[i + b->ndim - ndim];
    }

    uint64_t *c_indices = new uint64_t[ndim];
    std::fill(c_indices, c_indices + ndim, 0);

    *desc_ptr = new AddCpuDescriptor{
        DevCpu,
        c->dt,
        ndim,
        c_data_size,
        c->shape,
        a_strides,
        b_strides,
        c_indices,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyAddDescriptor(AddCpuDescriptor_t desc) {
    delete[] desc->a_strides;
    delete[] desc->b_strides;
    delete[] desc->c_indices;
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t add_cpu(AddCpuDescriptor_t desc, void *c, void const *a, void const *b) {
    auto a_ = reinterpret_cast<Tdata const *>(a);
    auto b_ = reinterpret_cast<Tdata const *>(b);
    auto c_ = reinterpret_cast<Tdata *>(c);
    const auto &indices = desc->c_indices;

    for (uint64_t i = 0; i < desc->c_data_size; ++i, incrementOne(indices, desc->c_shape, desc->ndim)) {
        auto a_index = compactToFlat(indices, desc->a_strides, desc->ndim);
        auto b_index = compactToFlat(indices, desc->b_strides, desc->ndim);
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            c_[i] = f32_to_f16(f16_to_f32(a_[a_index]) + f16_to_f32(b_[b_index]));
        } else {
            c_[i] = a_[a_index] + b_[b_index];
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuAdd(AddCpuDescriptor_t desc,
                        void *c, void const *a, void const *b,
                        void *stream) {
    if (desc->dtype == F16) {
        return add_cpu<uint16_t>(desc, c, a, b);
    }
    if (desc->dtype == F32) {
        return add_cpu<float>(desc, c, a, b);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
