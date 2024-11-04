#include "pooling_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreatePoolingDescriptor(infiniopHandle_t,
                                            PoolingCpuDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x,
                                            void const *kernel_shape,
                                            void const *pads,
                                            void const *strides,
                                            uint64_t n,
                                            int pooling_type) {
    uint64_t ndim = y->ndim;
    if (ndim < 3 || ndim != x->ndim || ndim != n + 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (x->shape[0] != y->shape[0] || x->shape[1] != y->shape[1]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (pooling_type > 1) {
        return STATUS_BAD_PARAM;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    const auto y_size = getTotalSize(y->shape, ndim);
    const auto pads_ = reinterpret_cast<uint64_t const *>(pads);
    const auto padded_x_size = requirePadding(pads_, ndim) ? getPaddedSize(ndim, x->shape, pads_) : 0;
    uint64_t *x_shape = new uint64_t[ndim];
    uint64_t *y_shape = new uint64_t[ndim];
    memcpy(x_shape, x->shape, ndim * sizeof(uint64_t));
    memcpy(y_shape, y->shape, ndim * sizeof(uint64_t));

    *desc_ptr = new PoolingCpuDescriptor{
        DevCpu,
        y->dt,
        ndim,
        y_size,
        padded_x_size,
        x_shape,
        reinterpret_cast<uint64_t const *>(kernel_shape),
        y_shape,
        reinterpret_cast<uint64_t const *>(pads),
        reinterpret_cast<int64_t const *>(strides),
        pooling_type,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGetPoolingWorkspaceSize(PoolingCpuDescriptor_t desc, uint64_t *size) {
    *size = desc->padded_x_size * desc->dt.size;
    if (desc->dt == F16) {
        *size += desc->y_size * sizeof(float);
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyPoolingDescriptor(PoolingCpuDescriptor_t desc) {
    delete[] desc->x_shape;
    delete[] desc->y_shape;
    delete desc;
    return STATUS_SUCCESS;
}

uint64_t getPaddedSize(uint64_t ndim, uint64_t *shape, uint64_t const *pads) {
    uint64_t total_size = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_size *= shape[i] + (i < 2 ? 0 : 2 * pads[i - 2]);
    }
    return total_size;
}

// calculate the padded shape and store the result in padded_shape
void getPaddedShape(uint64_t ndim, uint64_t const *shape, uint64_t const *pads, uint64_t *padded_shape) {
    memcpy(padded_shape, shape, ndim * sizeof(uint64_t));
    for (size_t i = 2; i < ndim; ++i) {
        padded_shape[i] += 2 * pads[i - 2];
    }
}

// initialize the padded input with the data from the original input
template<typename Tdata>
void fillPaddedInput(PoolingCpuDescriptor_t desc, uint64_t const *padded_x_shape,
                     Tdata *padded_x, Tdata const *x,
                     uint64_t const *pads, uint64_t x_index,
                     uint64_t padded_x_index, uint64_t ndim) {
    const auto x_shape = desc->x_shape[ndim];
    const auto padded_x_shape_ = padded_x_shape[ndim];
    const auto x_base_index = x_index * x_shape;
    const auto padded_x_base_index = padded_x_index * padded_x_shape_ +
                                     (x_shape == padded_x_shape_ ? 0 : pads[ndim - 2]);

    for (size_t i = 0; i < x_shape; ++i) {
        // base case (last dimension)
        if (ndim == desc->ndim - 1) {
            padded_x[padded_x_base_index + i] = x[x_base_index + i];
        }
        // recursive case
        else {
            fillPaddedInput(desc, padded_x_shape, padded_x, x, pads, x_base_index + i,
                            padded_x_base_index + i, ndim + 1);
        }
    }
}

// perform the a singleton pooling operation depending on the data type and pooling type
template<typename Xdata, typename Ydata>
inline void pool(PoolingCpuDescriptor_t desc, Ydata *y, Xdata const *x,
                 uint64_t const *x_shape, uint64_t curr_x_index, uint64_t y_index) {
    switch (desc->pooling_mode) {
        // 0. Max pooling
        case 0:
            if constexpr (std::is_same<Xdata, uint16_t>::value) {
                y[y_index] = std::fmax(f16_to_f32(x[curr_x_index]), y[y_index]);
            } else {
                y[y_index] = std::max(x[curr_x_index], y[y_index]);
            }
            break;
        // 1. Average pooling
        default:
            if constexpr (std::is_same<Xdata, uint16_t>::value) {
                y[y_index] += f16_to_f32(x[curr_x_index]);
            } else {
                y[y_index] += x[curr_x_index];
            }
    }
}

// Recursive convolution function
template<typename Xdata, typename Ydata>
void _applyPooling(PoolingCpuDescriptor_t desc, Ydata *y, Xdata const *x,
                   uint64_t const *x_shape, uint64_t x_index, uint64_t y_index,
                   uint64_t ndim) {
    const auto dim_size = x_shape[ndim];
    const auto kernel_size = desc->k_shape[ndim - 2];
    const auto dilation = 1;
    const auto stride = desc->strides[ndim - 2];
    const auto steps =
        (dim_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    x_index *= dim_size;
    y_index *= desc->y_shape[ndim];

    // perform all the pooling along this axis
    for (size_t i = 0; i < steps; ++i, ++y_index) {
        // perform a single pooling
        for (size_t k = 0; k < kernel_size; ++k) {
            // calculate the current indices
            const auto curr_x_index = x_index + i * stride + k * dilation;

            // base case (last dimension)
            if (ndim == desc->ndim - 1) {
                pool(desc, y, x, x_shape, curr_x_index, y_index);
            }
            // recursive case
            else {
                _applyPooling(desc, y, x, x_shape, curr_x_index, y_index, ndim + 1);
            }
        }
    }
}

template<typename Xdata, typename Ydata>
void applyPooling(PoolingCpuDescriptor_t desc, Ydata *y, Xdata const *x, uint64_t const *x_shape) {
#pragma omp parallel for
    // batch
    for (size_t i = 0; i < x_shape[0]; ++i) {
#pragma omp parallel for
        // channel
        for (size_t j = 0; j < x_shape[1]; ++j) {
            uint64_t x_index = i * x_shape[1] + j;
            uint64_t y_index = i * desc->y_shape[1] + j;
            _applyPooling(desc, y, x, x_shape, x_index, y_index, 2);
        }
    }

    // if is average pooling, take the average
    if (desc->pooling_mode == 1) {
        Ydata num_kernel_elements = getTotalSize(desc->k_shape, desc->ndim - 2);
#pragma omp parallel for
        for (size_t i = 0; i < desc->y_size; ++i) {
            y[i] /= num_kernel_elements;
        }
    }
}

template<typename Xdata, typename Ydata>
void _pooling_cpu(PoolingCpuDescriptor_t desc, void *workspace, uint64_t workspace_size,
                  Ydata *y, Xdata const *x) {
    if (desc->padded_x_size > 0) {
        auto padded_x = reinterpret_cast<Xdata *>(workspace);
        uint64_t padded_shape[desc->ndim];
        std::fill(padded_x, padded_x + desc->padded_x_size, 0);
        getPaddedShape(desc->ndim, desc->x_shape, desc->pads, padded_shape);
        fillPaddedInput<Xdata>(desc, padded_shape, padded_x, x, desc->pads, 0, 0, 0);
        applyPooling<Xdata, Ydata>(desc, y, padded_x, padded_shape);
    } else {
        applyPooling<Xdata, Ydata>(desc, y, x, desc->x_shape);
    }
}

// Pooling function
template<typename Tdata>
infiniopStatus_t pooling_cpu(PoolingCpuDescriptor_t desc, void *workspace, uint64_t workspace_size,
                             void *y, void const *x) {
    auto y_ = reinterpret_cast<Tdata *>(y);
    auto x_ = reinterpret_cast<Tdata const *>(x);
    std::fill(y_, y_ + desc->y_size, 0);
    _pooling_cpu<Tdata, Tdata>(desc, workspace, workspace_size, y_, x_);
    return STATUS_SUCCESS;
}

// sepcial case for fp16 (uint16_t)
template<>
infiniopStatus_t pooling_cpu<uint16_t>(PoolingCpuDescriptor_t desc, void *workspace, uint64_t workspace_size,
                                       void *y, void const *x) {
    auto y_ = reinterpret_cast<float *>(workspace);
    auto x_ = reinterpret_cast<uint16_t const *>(x);
    std::fill(y_, y_ + desc->y_size, 0);

    _pooling_cpu<uint16_t, float>(desc, y_ + desc->y_size, workspace_size, y_, x_);

    // copy data from y_ to y
    auto y_16 = reinterpret_cast<uint16_t *>(y);
    copyF32DataToF16(y_16, y_, desc->y_size);
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuPooling(PoolingCpuDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *y,
                            void const *x,
                            void *stream) {
    if (desc->dt == F16) {
        return pooling_cpu<uint16_t>(desc, workspace, workspace_size, y, x);
    }
    if (desc->dt == F32) {
        return pooling_cpu<float>(desc, workspace, workspace_size, y, x);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
