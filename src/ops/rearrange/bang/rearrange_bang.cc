#include "rearrange_bang.h"
#include "../../../devices/bang/common_bang.h"
#include "../../utils.h"
#include <numeric>

infiniopStatus_t bangCreateRearrangeDescriptor(BangHandle_t handle,
                                               RearrangeBangDescriptor_t *desc_ptr,
                                               infiniopTensorDescriptor_t dst,
                                               infiniopTensorDescriptor_t src) {
    if (!dtype_eq(dst->dt, src->dt)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (dst->ndim != src->ndim || dst->ndim < 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    auto ndim = dst->ndim;
    for (size_t i = 0; i < ndim; ++i) {
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
        for (size_t i = ndim - 3; i >= 1; --i) {
            if (static_cast<uint64_t>(dst->shape[i]) * static_cast<uint64_t>(dst->strides[i]) != static_cast<uint64_t>(dst->strides[i - 1]) ||
                static_cast<uint64_t>(src->shape[i]) * static_cast<uint64_t>(src->strides[i]) != static_cast<uint64_t>(src->strides[i - 1])) {
                return STATUS_BAD_TENSOR_STRIDES;
            }
        }
        r = std::accumulate(dst->shape, dst->shape + ndim - 1, 1, std::multiplies<unsigned int>());
    }
    char *tmpDevice;
    CNRT_CHECK(cnrtMalloc((void **) &tmpDevice, ndim * sizeof(uint64_t) + 2 * ndim * sizeof(int64_t)));
    char *mlu_stride = tmpDevice + ndim * sizeof(uint64_t);
    uint64_t *mlu_shape = (uint64_t *) tmpDevice;

    int64_t *mlu_strides_dst = (int64_t *) mlu_stride;
    int64_t *mlu_strides_src = mlu_strides_dst + ndim;


    CNRT_CHECK(cnrtMemcpy(mlu_shape, dst->shape, ndim * sizeof(uint64_t), cnrtMemcpyHostToDev));

    CNRT_CHECK(cnrtMemcpy(mlu_strides_dst, dst->strides, ndim * sizeof(int64_t), cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpy(mlu_strides_src, src->strides, ndim * sizeof(int64_t), cnrtMemcpyHostToDev));
    *desc_ptr = new RearrangeBangDescriptor{
        handle->device,
        handle->device_id,
        dst->dt,
        r,
        ndim,
        mlu_shape,
        mlu_strides_dst, mlu_strides_src};
    return STATUS_SUCCESS;
}
infiniopStatus_t bangDestroyRearrangeDescriptor(RearrangeBangDescriptor_t desc) {
    cnrtFree(desc->mlu_shape);

    delete desc;
    return STATUS_SUCCESS;
}
