#include "../unary/unary.h"
#include "../utils.h"
#include "ops/sign/sign.h"
#include "tensor/tensor_descriptor.h"

struct _SignDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _SignDescriptor *_SignDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSignDescriptor(infiniopHandle_t handle,
                                                           infiniopSignDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y_desc,
                                                           infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Sign), STATUS_SUCCESS);

    *(_SignDescriptor_t *) desc_ptr = new _SignDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopSign(infiniopSignDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream) {
    auto _desc = (_SignDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroySignDescriptor(infiniopSignDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_SignDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}