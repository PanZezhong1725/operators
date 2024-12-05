#include "../unary/unary.h"
#include "../utils.h"
#include "ops/tan/tan.h"
#include "tensor/tensor_descriptor.h"

struct _TanDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _TanDescriptor *_TanDescriptor_t;

__C __export infiniopStatus_t infiniopCreateTanDescriptor(infiniopHandle_t handle,
                                                          infiniopTanDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y_desc,
                                                          infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Tan), STATUS_SUCCESS);

    *(_TanDescriptor_t *) desc_ptr = new _TanDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopTan(infiniopTanDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream) {
    auto _desc = (_TanDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyTanDescriptor(infiniopTanDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_TanDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}