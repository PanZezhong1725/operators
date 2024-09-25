#include "conv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

// get the total number of elements in arr
inline uint64_t getTotalSize(const uint64_t *arr, uint64_t ndim) {
    return std::accumulate(arr, arr + ndim, 1ULL, std::multiplies<uint64_t>());
}

// check if padding is needed
inline bool requirePadding(uint64_t const *pads, uint64_t ndim) {
    return std::any_of(pads, pads + ndim - 2,
                       [](uint64_t pad) { return pad > 0; });
}

/**
 * get the total array size (element count) after applying padding for a 
 * ndim-ary tensor with the given shape
 */
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

infiniopStatus_t cpuCreateConvDescriptor(infiniopHandle_t,
                                         ConvCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t y,
                                         infiniopTensorDescriptor_t x,
                                         infiniopTensorDescriptor_t w,
                                         void const *pads,
                                         void const *strides,
                                         void const *dilations,
                                         uint64_t n) {
    uint64_t ndim = y->ndim;
    if (ndim < 3 || ndim != x->ndim || ndim != w->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (x->shape[0] != y->shape[0] || w->shape[0] != y->shape[1] || x->shape[1] != w->shape[1]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (y->dt != F16 || y->dt != x->dt || y->dt != w->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t y_size = getTotalSize(y->shape, ndim);
    const auto pads_ = reinterpret_cast<uint64_t const *>(pads);
    uint64_t padded_x_size = requirePadding(pads_, ndim) ? getPaddedSize(ndim, x->shape, pads_) : 0;

    *desc_ptr = new ConvCpuDescriptor{
        DevCpu,
        y->dt,
        ndim,
        y_size,
        padded_x_size,
        x->shape,
        w->shape,
        y->shape,
        reinterpret_cast<uint64_t const *>(pads),
        reinterpret_cast<int64_t const *>(strides),
        reinterpret_cast<uint64_t const *>(dilations),
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGetConvWorkspaceSize(ConvCpuDescriptor_t desc, uint64_t *size) {
    *size = desc->y_size * sizeof(float) + desc->padded_x_size * sizeof(uint16_t);
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyConvDescriptor(ConvCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

// copy the data in src tensor into that of the dest tensor but also convert
// from f32 to f16
void copyF32DataToF16(uint16_t *dest, float const *src, uint64_t size) {
    for (size_t i = 0; i < size; ++i) {
        dest[i] = f32_to_f16(src[i]);
    }
}

// initialize the padded input with the data from the original input
void fillPaddedInput(ConvCpuDescriptor_t desc, uint64_t const *padded_x_shape,
                     uint16_t *padded_x, uint16_t const *x,
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

// Recursive convolution function
void _applyConv(ConvCpuDescriptor_t desc, float *y, uint16_t const *x,
                uint16_t const *w, uint64_t const *x_shape,
                uint64_t x_index, uint64_t w_index, uint64_t y_index,
                uint64_t ndim) {
    const auto dim_size = x_shape[ndim];
    const auto kernel_size = desc->w_shape[ndim];
    const auto dilation = desc->dilations[ndim - 2];
    const auto stride = desc->strides[ndim - 2];
    const auto steps =
        (dim_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    x_index *= dim_size;
    w_index *= kernel_size;
    y_index *= desc->y_shape[ndim];

    // perform all the convolutions along this axis
    for (size_t i = 0; i < steps; ++i, ++y_index) {
        // perform a single convolution
        for (size_t k = 0; k < kernel_size; ++k) {
            // calculate the current indices
            const auto curr_x_index = x_index + i * stride + k * dilation;
            const auto curr_w_index = w_index + k;

            // base case (last dimension)
            if (ndim == desc->ndim - 1) {
                y[y_index] += f16_to_f32(x[curr_x_index]) * f16_to_f32(w[curr_w_index]);
            }
            // recursive case
            else {
                _applyConv(desc, y, x, w, x_shape, curr_x_index, curr_w_index,
                           y_index, ndim + 1);
            }
        }
    }
}

void applyConv(ConvCpuDescriptor_t desc, float *y, uint16_t const *x,
               uint16_t const *w, uint64_t const *x_shape) {
    const auto y_num_channel_elements =
        getTotalSize(desc->y_shape + 2, desc->ndim - 2);

    // batch
    for (size_t i = 0; i < x_shape[0]; ++i) {

        // output channel
        for (size_t j = 0; j < desc->w_shape[0]; ++j) {
            uint64_t y_index = i * desc->y_shape[1] + j;

            // input channel
            for (size_t k = 0; k < x_shape[1]; ++k) {
                uint64_t x_index = i * x_shape[1] + k;
                uint64_t w_index = j * desc->w_shape[1] + k;
                _applyConv(desc, y, x, w, x_shape, x_index, w_index, y_index, 2);
            }
        }
    }
}

// Convolution function
void conv_cpu_f16(ConvCpuDescriptor_t desc, void *workspace, uint64_t workspace_size,
                  void *y, void const *x, void const *w) {
    auto y_ = reinterpret_cast<float *>(workspace);
    auto x_ = reinterpret_cast<uint16_t const *>(x);
    auto w_ = reinterpret_cast<uint16_t const *>(w);
    std::fill(y_, y_ + desc->y_size, 0);

    if (desc->padded_x_size > 0) {
        auto padded_x = reinterpret_cast<uint16_t *>(y_ + desc->y_size);
        uint64_t padded_shape[desc->ndim];
        std::fill(padded_x, padded_x + desc->padded_x_size, 0);
        getPaddedShape(desc->ndim, desc->x_shape, desc->pads, padded_shape);
        fillPaddedInput(desc, padded_shape, padded_x, x_, desc->pads, 0, 0, 0);
        applyConv(desc, y_, padded_x, w_, padded_shape);
    } else {
        applyConv(desc, y_, x_, w_, desc->x_shape);
    }

    // copy data from y_ to y
    auto y_16 = reinterpret_cast<uint16_t *>(y);
    copyF32DataToF16(y_16, y_, desc->y_size);
}

infiniopStatus_t cpuConv(ConvCpuDescriptor_t desc,
                         void *workspace, uint64_t workspace_size,
                         void *y, void const *x, void const *w,
                         void *stream) {
    if (desc->dtype == F16) {
        conv_cpu_f16(desc, workspace, workspace_size, y, x, w);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}