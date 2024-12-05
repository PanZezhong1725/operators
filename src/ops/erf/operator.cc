#include "../unary/unary.h"
#include "../utils.h"
#include "ops/erf/erf.h"
#include "tensor/tensor_descriptor.h"

struct _ErfDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _ErfDescriptor *_ErfDescriptor_t;

__C __export infiniopStatus_t infiniopCreateErfDescriptor(infiniopHandle_t handle,
                                                          infiniopErfDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y_desc,
                                                          infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Erf), STATUS_SUCCESS);

    *(_ErfDescriptor_t *) desc_ptr = new _ErfDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopErf(infiniopErfDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream) {
    auto _desc = (_ErfDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyErfDescriptor(infiniopErfDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_ErfDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}