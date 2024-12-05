#include "../unary/unary.h"
#include "../utils.h"
#include "ops/cos/cos.h"
#include "tensor/tensor_descriptor.h"

struct _CosDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _CosDescriptor *_CosDescriptor_t;

__C __export infiniopStatus_t infiniopCreateCosDescriptor(infiniopHandle_t handle,
                                                          infiniopCosDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y_desc,
                                                          infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Cos), STATUS_SUCCESS);

    *(_CosDescriptor_t *) desc_ptr = new _CosDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopCos(infiniopCosDescriptor_t desc,
                                          void *y,
                                          void const *x,
                                          void *stream) {
    auto _desc = (_CosDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyCosDescriptor(infiniopCosDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_CosDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}