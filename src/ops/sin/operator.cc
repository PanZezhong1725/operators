#include "../unary/unary.h"
#include "../utils.h"
#include "ops/sin/sin.h"
#include "tensor/tensor_descriptor.h"

struct _SinDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _SinDescriptor *_SinDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSinDescriptor(infiniopHandle_t handle,
                                                          infiniopSinDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y_desc,
                                                          infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Sin), STATUS_SUCCESS);

    *(_SinDescriptor_t *) desc_ptr = new _SinDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopSin(infiniopSinDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream) {
    auto _desc = (_SinDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroySinDescriptor(infiniopSinDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_SinDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}