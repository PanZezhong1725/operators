#include "rearrange_cpu.h"
#include "../../utils.h"
#include <cstdint>
#include <cstring>
#include <numeric>

infiniopStatus_t cpuCreateRearrangeDescriptor(infiniopHandle_t,
                                              RearrangeCpuDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t dst,
                                              infiniopTensorDescriptor_t src) {
    if (!dtype_eq(dst->dt, src->dt)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    auto ndim = dst->ndim;
    if (src->ndim != ndim || ndim == 0) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (int i = 0; i < ndim; ++i) {
        if (dst->shape[i] != src->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (dst->strides[ndim - 1] != 1 || src->strides[ndim - 1] != 1) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    std::vector<uint64_t>
        shape(dst->shape, dst->shape + ndim);
    std::vector<int64_t>
        strides_dst(dst->strides, dst->strides + ndim),
        strides_src(src->strides, src->strides + ndim);

    unsigned int r = 0;
    switch (ndim) {
        case 1:
            ndim = 2;
            strides_dst.insert(strides_dst.begin(), shape[0]);
            strides_src.insert(strides_src.begin(), shape[0]);
            shape.insert(shape.begin(), 1);
        case 2:
            r = shape[0];
            break;
        case 3:
            r = shape[0] * shape[1];
            break;
        default:
            for (int i = ndim - 3; i >= 1; --i) {
                if (shape[i] * strides_dst[i] != strides_dst[i - 1] || shape[i] * strides_src[i] != strides_src[i - 1]) {
                    return STATUS_BAD_TENSOR_STRIDES;
                }
            }
            r = std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies{});
            break;
    }
    *desc_ptr = new RearrangeCpuDescriptor{
        DevCpu,
        dst->dt,
        r,
        shape,
        strides_dst,
        strides_src,
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyRearrangeDescriptor(RearrangeCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

inline int indices(uint64_t i, uint64_t ndim, std::vector<int64_t> strides, std::vector<uint64_t> shape) {
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
    auto ndim = desc->shape.size();
    int bytes_size = desc->shape[ndim - 1] * desc->dt.size;
#pragma omp parallel for
    for (uint64_t i = 0; i < desc->r; ++i) {
        auto dst_offset = indices(i, ndim, desc->strides_dst, desc->shape);
        auto src_offset = indices(i, ndim, desc->strides_src, desc->shape);
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
