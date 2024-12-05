#include "../unary/unary.h"
#include "../utils.h"
#include "ops/exp/exp.h"
#include "tensor/tensor_descriptor.h"

struct _ExpDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _ExpDescriptor *_ExpDescriptor_t;

__C __export infiniopStatus_t infiniopCreateExpDescriptor(infiniopHandle_t handle,
                                                          infiniopExpDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y_desc,
                                                          infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Exp), STATUS_SUCCESS);

    *(_ExpDescriptor_t *) desc_ptr = new _ExpDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopExp(infiniopExpDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream) {
    auto _desc = (_ExpDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyExpDescriptor(infiniopExpDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_ExpDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}