#include "../unary/unary.h"
#include "../utils.h"
#include "ops/cosh/cosh.h"
#include "tensor/tensor_descriptor.h"

struct _CoshDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _CoshDescriptor *_CoshDescriptor_t;

__C __export infiniopStatus_t infiniopCreateCoshDescriptor(infiniopHandle_t handle,
                                                           infiniopCoshDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y_desc,
                                                           infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Cosh), STATUS_SUCCESS);

    *(_CoshDescriptor_t *) desc_ptr = new _CoshDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopCosh(infiniopCoshDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream) {
    auto _desc = (_CoshDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyCoshDescriptor(infiniopCoshDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_CoshDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}