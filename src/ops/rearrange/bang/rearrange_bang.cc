#include "rearrange_bang.h"
#include "../../../devices/bang/common_bang.h"
#include "../../utils.h"
#include <numeric>

infiniopStatus_t bangCreateRearrangeDescriptor(BangHandle_t handle,
                                               RearrangeBangDescriptor_t *desc_ptr,
                                               infiniopTensorDescriptor_t dst,
                                               infiniopTensorDescriptor_t src) {
    auto dt = dst->dt;
    if (!dtype_eq(src->dt, dt)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    auto ndim = dst->ndim;
    if (src->ndim != ndim || ndim == 0) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (decltype(ndim) i = 0; i < ndim; ++i) {
        if (dst->shape[i] != src->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (dst->strides[ndim - 1] != 1 || src->strides[ndim - 1] != 1) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    unsigned int r = 0;
    std::vector<uint64_t> shape_;
    std::vector<int64_t> dst_strides, src_strides;
    switch (ndim) {
        case 1:
            shape_.push_back(dst->shape[0]);
            dst_strides.push_back(0);
            src_strides.push_back(0);
            r = 1;
            break;
        case 2:
            r = dst->shape[0];
            break;
        case 3:
            r = dst->shape[0] * dst->shape[1];
            break;
        default: {
            for (size_t i = ndim - 3; i >= 1; --i) {
                if (static_cast<uint64_t>(dst->shape[i]) * static_cast<uint64_t>(dst->strides[i]) != static_cast<uint64_t>(dst->strides[i - 1]) ||
                    static_cast<uint64_t>(src->shape[i]) * static_cast<uint64_t>(src->strides[i]) != static_cast<uint64_t>(src->strides[i - 1])) {
                    return STATUS_BAD_TENSOR_STRIDES;
                }
            }
            r = std::accumulate(dst->shape, dst->shape + ndim - 1, 1, std::multiplies<unsigned int>());
            break;
        }
    }

    for (decltype(ndim) i = 0; i < ndim; ++i) {
        shape_.push_back(dst->shape[i]);
        dst_strides.push_back(dst->strides[i]);
        src_strides.push_back(src->strides[i]);
    }

    char *tmpDevice;
    CNRT_CHECK(cnrtMalloc((void **) &tmpDevice, ndim * sizeof(uint64_t) + 2 * ndim * sizeof(int64_t)));
    char *mlu_stride = tmpDevice + ndim * sizeof(uint64_t);
    uint64_t *mlu_shape = (uint64_t *) tmpDevice;

    int64_t *mlu_strides_dst = (int64_t *) mlu_stride;
    int64_t *mlu_strides_src = mlu_strides_dst + ndim;

    CNRT_CHECK(cnrtMemcpy(mlu_shape, shape_.data(), ndim * sizeof(uint64_t), cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpy(mlu_strides_dst, dst_strides.data(), ndim * sizeof(int64_t), cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpy(mlu_strides_src, src_strides.data(), ndim * sizeof(int64_t), cnrtMemcpyHostToDev));
    *desc_ptr = new RearrangeBangDescriptor{
        handle->device,
        handle->device_id,
        dst->dt,
        r,
        ndim,
        mlu_shape,
        mlu_strides_dst,
        mlu_strides_src};
    return STATUS_SUCCESS;
}
infiniopStatus_t bangDestroyRearrangeDescriptor(RearrangeBangDescriptor_t desc) {
    cnrtFree(desc->mlu_shape);

    delete desc;
    return STATUS_SUCCESS;
}
