#include "../unary/unary.h"
#include "../utils.h"
#include "ops/acos/acos.h"
#include "tensor/tensor_descriptor.h"

struct _AcosDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _AcosDescriptor *_AcosDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAcosDescriptor(infiniopHandle_t handle,
                                                           infiniopAcosDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y_desc,
                                                           infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Acos), STATUS_SUCCESS);

    *(_AcosDescriptor_t *) desc_ptr = new _AcosDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopAcos(infiniopAcosDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream) {
    auto _desc = (_AcosDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyAcosDescriptor(infiniopAcosDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_AcosDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}