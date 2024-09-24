#include "rearrange_cpu.h"
#include "../../utils.h"
#include <cstring>
#include <numeric>

infiniopStatus_t cpuCreateRearrangeDescriptor(infiniopHandle_t,
                                              RearrangeCpuDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t dst,
                                              infiniopTensorDescriptor_t src) {
    if (!dtype_eq(dst->dt, src->dt)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (dst->ndim != src->ndim || dst->ndim < 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    auto ndim = dst->ndim;
    for (int i = 0; i < ndim; ++i) {
        if (dst->shape[i] != src->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (dst->strides[ndim - 1] != 1 || src->strides[ndim - 1] != 1) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    unsigned int r = 0;
    if (ndim == 2) {
        r = dst->shape[0];
    } else if (ndim == 3) {
        r = dst->shape[0] * dst->shape[1];
    } else {
        for (int i = ndim - 3; i >= 1; --i) {
            if (dst->shape[i] * dst->strides[i] != dst->strides[i - 1] || src->shape[i] * src->strides[i] != src->strides[i - 1]) {
                return STATUS_BAD_TENSOR_STRIDES;
            }
        }
        r = std::accumulate(dst->shape, dst->shape + ndim - 1, 1, std::multiplies<unsigned int>());
    }
    *desc_ptr = new RearrangeCpuDescriptor{
        DevCpu,
        dst->dt,
        r,
        ndim,
        dst->shape, src->shape,
        dst->strides, src->strides};
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyRearrangeDescriptor(RearrangeCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

inline int indices(uint64_t i, uint64_t ndim, int64_t *strides, uint64_t *shape) {
    uint64_t ans = 0;
    for (int j = ndim - 2; j >= 0; --j) {
        ans += (i % shape[j]) * strides[j];
        i /= shape[j];
    }
    return ans;
}

void reform_cpu(RearrangeCpuDescriptor_t desc, void *dst, void const *src) {
    auto dst_ptr = reinterpret_cast<uint8_t *>(dst);
    auto src_ptr = reinterpret_cast<const uint8_t *>(src);
    int bytes_size = desc->shape_dst[desc->ndim - 1] * desc->dt.size;
#pragma omp parallel for
    for (uint64_t i = 0; i < desc->r; ++i) {
        auto dst_offset = indices(i, desc->ndim, desc->strides_dst, desc->shape_dst);
        auto src_offset = indices(i, desc->ndim, desc->strides_src, desc->shape_src);
        std::memcpy(dst_ptr + dst_offset * desc->dt.size, src_ptr + src_offset * desc->dt.size, bytes_size);
    }
}

infiniopStatus_t cpuRearrange(RearrangeCpuDescriptor_t desc,
                              void *dst,
                              void const *src,
                              void *stream) {
    reform_cpu(desc, dst, src);
    return STATUS_SUCCESS;
}
