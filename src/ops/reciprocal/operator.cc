#include "../unary/unary.h"
#include "../utils.h"
#include "ops/reciprocal/reciprocal.h"
#include "tensor/tensor_descriptor.h"

struct _ReciprocalDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};
typedef struct _ReciprocalDescriptor *_ReciprocalDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReciprocalDescriptor(infiniopHandle_t handle,
                                                                 infiniopReciprocalDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t y_desc,
                                                                 infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Reciprocal), STATUS_SUCCESS);

    *(_ReciprocalDescriptor_t *) desc_ptr = new _ReciprocalDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopReciprocal(infiniopReciprocalDescriptor_t desc,
                                                 void *y,
                                                 void const *x,
                                                 void *stream) {
    auto _desc = (_ReciprocalDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyReciprocalDescriptor(infiniopReciprocalDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_ReciprocalDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}