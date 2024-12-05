#include "../unary/unary.h"
#include "../utils.h"
#include "ops/acosh/acosh.h"
#include "tensor/tensor_descriptor.h"

struct _AcoshDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _AcoshDescriptor *_AcoshDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAcoshDescriptor(infiniopHandle_t handle,
                                                            infiniopAcoshDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y_desc,
                                                            infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Acosh), STATUS_SUCCESS);

    *(_AcoshDescriptor_t *) desc_ptr = new _AcoshDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopAcosh(infiniopAcoshDescriptor_t desc,
                                            void *y,
                                            void const *x,
                                            void *stream) {
    auto _desc = (_AcoshDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyAcoshDescriptor(infiniopAcoshDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_AcoshDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}