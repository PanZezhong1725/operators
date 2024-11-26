#include "transpose_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateTransposeDescriptor(infiniopHandle_t,
                                              TransposeCpuDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t y,
                                              infiniopTensorDescriptor_t x,
                                              uint64_t const *perm,
                                              uint64_t n) {
    uint64_t ndim = y->ndim;
    if (ndim != x->ndim || ndim != n) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t y_data_size = std::accumulate(y->shape, y->shape + y->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t x_data_size = std::accumulate(x->shape, x->shape + x->ndim, 1ULL, std::multiplies<uint64_t>());
    if (y_data_size != x_data_size) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    infiniopTensorDescriptor_t transposed_tensor = permute(x, {perm, perm + n});
    if (!transposed_tensor) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (transposed_tensor->shape[i] != y->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }

    TransposeMode mode = TransposeMode::NON_CONTIGUOUS_COPY;
    bool y_is_contiguous = is_contiguous(y);
    bool transposed = false;

    // determine whether the tranpose operation can be directly collapsed into copying
    if (ndim < 2 || can_squeeze_to_1D(x->shape, ndim) || is_same(perm, n)) {
        mode = (y_is_contiguous && is_contiguous(x)) ? TransposeMode::FULL_CONTIGUOUS_COPY : y_is_contiguous ? TransposeMode::OUTPUT_CONTIGUOUS_COPY
                                                                                                             : TransposeMode::NON_CONTIGUOUS_COPY;
    } else {
        transposed = true;
        mode = y_is_contiguous ? TransposeMode::OUTPUT_CONTIGUOUS_COPY : TransposeMode::NON_CONTIGUOUS_COPY;
    }

    // if perm is not provided, by default, tranpose should reverse the dimensions
    if (!perm && !are_reverse(x->shape, y->shape, ndim)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    int64_t *x_strides = new int64_t[ndim];
    int64_t *y_strides = new int64_t[ndim];
    uint64_t *x_shape = new uint64_t[ndim];
    uint64_t *y_shape = new uint64_t[ndim];
    memcpy(x_shape, transposed ? transposed_tensor->shape : x->shape, ndim * sizeof(uint64_t));
    memcpy(y_shape, y->shape, ndim * sizeof(uint64_t));
    memcpy(x_strides, transposed ? transposed_tensor->strides : x->strides, ndim * sizeof(int64_t));
    memcpy(y_strides, y->strides, ndim * sizeof(int64_t));

    *desc_ptr = new TransposeCpuDescriptor{
        DevCpu,
        y->dt,
        ndim,
        y_data_size,
        x_strides,
        y_strides,
        x_shape,
        y_shape,
        mode,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyTransposeDescriptor(TransposeCpuDescriptor_t desc) {
    delete[] desc->x_strides;
    delete[] desc->y_strides;
    delete[] desc->x_shape;
    delete[] desc->y_shape;
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t transpose_cpu(TransposeCpuDescriptor_t desc, void *y, void const *x) {
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);

#pragma omp parallel for
    for (uint64_t i = 0; i < desc->data_size; ++i) {
        switch (desc->mode) {
            case TransposeMode::FULL_CONTIGUOUS_COPY:
                y_[i] = x_[i];
                break;
            case TransposeMode::OUTPUT_CONTIGUOUS_COPY:
                y_[i] = x_[getOffset(i, desc->ndim, desc->x_shape, desc->x_strides)];
                break;
            default:// TransposeMode::NON_CONTIGUOUS_COPY:
                y_[getOffset(i, desc->ndim, desc->y_shape, desc->y_strides)] = x_[getOffset(i, desc->ndim, desc->x_shape, desc->x_strides)];
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuTranspose(TransposeCpuDescriptor_t desc,
                              void *y, void const *x,
                              void *stream) {
    if (desc->dtype == F16) {
        return transpose_cpu<uint16_t>(desc, y, x);
    }
    if (desc->dtype == F32) {
        return transpose_cpu<float>(desc, y, x);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
