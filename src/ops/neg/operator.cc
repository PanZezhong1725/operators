#include "../unary/unary.h"
#include "../utils.h"
#include "ops/neg/neg.h"
#include "tensor/tensor_descriptor.h"

struct _NegDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _NegDescriptor *_NegDescriptor_t;

__C __export infiniopStatus_t infiniopCreateNegDescriptor(infiniopHandle_t handle,
                                                          infiniopNegDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y_desc,
                                                          infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Neg), STATUS_SUCCESS);

    *(_NegDescriptor_t *) desc_ptr = new _NegDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopNeg(infiniopNegDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream) {
    auto _desc = (_NegDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyNegDescriptor(infiniopNegDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_NegDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}