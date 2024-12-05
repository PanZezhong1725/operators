#include "../unary/unary.h"
#include "../utils.h"
#include "ops/asin/asin.h"
#include "tensor/tensor_descriptor.h"

struct _AsinDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _AsinDescriptor *_AsinDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAsinDescriptor(infiniopHandle_t handle,
                                                           infiniopAsinDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y_desc,
                                                           infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Asin), STATUS_SUCCESS);

    *(_AsinDescriptor_t *) desc_ptr = new _AsinDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopAsin(infiniopAsinDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream) {
    auto _desc = (_AsinDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyAsinDescriptor(infiniopAsinDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_AsinDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}