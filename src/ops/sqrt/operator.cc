#include "../unary/unary.h"
#include "../utils.h"
#include "ops/sqrt/sqrt.h"
#include "tensor/tensor_descriptor.h"

struct _SqrtDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _SqrtDescriptor *_SqrtDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSqrtDescriptor(infiniopHandle_t handle,
                                                           infiniopSqrtDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y_desc,
                                                           infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Sqrt), STATUS_SUCCESS);

    *(_SqrtDescriptor_t *) desc_ptr = new _SqrtDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopSqrt(infiniopSqrtDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream) {
    auto _desc = (_SqrtDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroySqrtDescriptor(infiniopSqrtDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_SqrtDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}