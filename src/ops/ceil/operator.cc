#include "../unary/unary.h"
#include "../utils.h"
#include "ops/ceil/ceil.h"
#include "tensor/tensor_descriptor.h"

struct _CeilDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _CeilDescriptor *_CeilDescriptor_t;

__C __export infiniopStatus_t infiniopCreateCeilDescriptor(infiniopHandle_t handle,
                                                           infiniopCeilDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y_desc,
                                                           infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Ceil), STATUS_SUCCESS);

    *(_CeilDescriptor_t *) desc_ptr = new _CeilDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopCeil(infiniopCeilDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream) {
    auto _desc = (_CeilDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyCeilDescriptor(infiniopCeilDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_CeilDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}