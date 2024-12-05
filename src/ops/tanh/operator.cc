#include "../unary/unary.h"
#include "../utils.h"
#include "ops/tanh/tanh.h"
#include "tensor/tensor_descriptor.h"

struct _TanhDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _TanhDescriptor *_TanhDescriptor_t;

__C __export infiniopStatus_t infiniopCreateTanhDescriptor(infiniopHandle_t handle,
                                                           infiniopTanhDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y_desc,
                                                           infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Tanh), STATUS_SUCCESS);

    *(_TanhDescriptor_t *) desc_ptr = new _TanhDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopTanh(infiniopTanhDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream) {
    auto _desc = (_TanhDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyTanhDescriptor(infiniopTanhDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_TanhDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}