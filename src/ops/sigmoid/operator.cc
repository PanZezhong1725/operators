#include "../unary/unary.h"
#include "../utils.h"
#include "ops/sigmoid/sigmoid.h"
#include "tensor/tensor_descriptor.h"

struct _SigmoidDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _SigmoidDescriptor *_SigmoidDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSigmoidDescriptor(infiniopHandle_t handle,
                                                              infiniopSigmoidDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t y_desc,
                                                              infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Sigmoid), STATUS_SUCCESS);

    *(_SigmoidDescriptor_t *) desc_ptr = new _SigmoidDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopSigmoid(infiniopSigmoidDescriptor_t desc,
                                              void *y,
                                              void const *x,
                                              void *stream) {
    auto _desc = (_SigmoidDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroySigmoidDescriptor(infiniopSigmoidDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_SigmoidDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}