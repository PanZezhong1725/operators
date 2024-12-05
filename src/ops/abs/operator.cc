#include "../unary/unary.h"
#include "../utils.h"
#include "ops/abs/abs.h"
#include "tensor/tensor_descriptor.h"

struct _AbsDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _AbsDescriptor *_AbsDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAbsDescriptor(infiniopHandle_t handle,
                                                          infiniopAbsDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y_desc,
                                                          infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Abs), STATUS_SUCCESS);

    *(_AbsDescriptor_t *) desc_ptr = new _AbsDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopAbs(infiniopAbsDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream) {
    auto _desc = (_AbsDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyAbsDescriptor(infiniopAbsDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_AbsDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}