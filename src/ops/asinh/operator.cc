#include "../unary/unary.h"
#include "../utils.h"
#include "ops/asinh/asinh.h"
#include "tensor/tensor_descriptor.h"

struct _AsinhDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _AsinhDescriptor *_AsinhDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAsinhDescriptor(infiniopHandle_t handle,
                                                            infiniopAsinhDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y_desc,
                                                            infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Asinh), STATUS_SUCCESS);

    *(_AsinhDescriptor_t *) desc_ptr = new _AsinhDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopAsinh(infiniopAsinhDescriptor_t desc,
                                            void *y,
                                            void const *x,
                                            void *stream) {
    auto _desc = (_AsinhDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyAsinhDescriptor(infiniopAsinhDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_AsinhDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}