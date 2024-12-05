#include "../unary/unary.h"
#include "../utils.h"
#include "ops/atan/atan.h"
#include "tensor/tensor_descriptor.h"

struct _AtanDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _AtanDescriptor *_AtanDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAtanDescriptor(infiniopHandle_t handle,
                                                           infiniopAtanDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y_desc,
                                                           infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Atan), STATUS_SUCCESS);

    *(_AtanDescriptor_t *) desc_ptr = new _AtanDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopAtan(infiniopAtanDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream) {
    auto _desc = (_AtanDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyAtanDescriptor(infiniopAtanDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_AtanDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}