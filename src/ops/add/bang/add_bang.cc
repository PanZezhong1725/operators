#include "add_bang.h"
#include "../../../devices/bang/common_bang.h"
#include "../../utils.h"
#include <cmath>
infiniopStatus_t bangCreateAddDescriptor(BangHandle_t handle,
                                         AddBangDescriptor_t *desc_ptr,
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
    if (!dtype_eq(c->dt, F16) || c->dt != a->dt || c->dt != b->dt) {
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
    uint64_t *c_shape, *a_strides_d, *b_strides_d;
    cnrtMalloc((void **) &c_shape, ndim * sizeof(uint64_t));
    cnrtMalloc((void **) &a_strides_d, ndim * sizeof(uint64_t));
    cnrtMalloc((void **) &b_strides_d, ndim * sizeof(uint64_t));
    cnrtMemcpy(c_shape, c->shape, ndim * sizeof(uint64_t), cnrtMemcpyHostToDev);
    cnrtMemcpy(a_strides_d, a_strides, ndim * sizeof(uint64_t), cnrtMemcpyHostToDev);
    cnrtMemcpy(b_strides_d, b_strides, ndim * sizeof(uint64_t), cnrtMemcpyHostToDev);
    *desc_ptr = new AddBangDescriptor{
        handle->device,
        handle->device_id,
        c->dt,
        ndim,
        c_data_size,
        c_shape,
        a_strides_d,
        b_strides_d};

    return STATUS_SUCCESS;
}

infiniopStatus_t bangDestroyAddDescriptor(AddBangDescriptor_t desc) {
    cnrtFree(desc->c_shape);
    cnrtFree(desc->a_strides_d);
    cnrtFree(desc->b_strides_d);
    delete desc;
    return STATUS_SUCCESS;
}
