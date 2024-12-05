#include "../unary/unary.h"
#include "../utils.h"
#include "ops/log/log.h"
#include "tensor/tensor_descriptor.h"

struct _LogDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _LogDescriptor *_LogDescriptor_t;

__C __export infiniopStatus_t infiniopCreateLogDescriptor(infiniopHandle_t handle,
                                                          infiniopLogDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y_desc,
                                                          infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Log), STATUS_SUCCESS);

    *(_LogDescriptor_t *) desc_ptr = new _LogDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopLog(infiniopLogDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream) {
    auto _desc = (_LogDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyLogDescriptor(infiniopLogDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_LogDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}