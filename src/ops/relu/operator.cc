#include "../unary/unary.h"
#include "../utils.h"
#include "ops/relu/relu.h"
#include "tensor/tensor_descriptor.h"

struct _ReluDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _ReluDescriptor *_ReluDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReluDescriptor(infiniopHandle_t handle,
                                                           infiniopReluDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y_desc,
                                                           infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Relu), STATUS_SUCCESS);

    *(_ReluDescriptor_t *) desc_ptr = new _ReluDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopRelu(infiniopReluDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream) {
    auto _desc = (_ReluDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyReluDescriptor(infiniopReluDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_ReluDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}