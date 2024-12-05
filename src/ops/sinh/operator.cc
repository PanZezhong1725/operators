#include "../unary/unary.h"
#include "../utils.h"
#include "ops/sinh/sinh.h"
#include "tensor/tensor_descriptor.h"

struct _SinhDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _SinhDescriptor *_SinhDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSinhDescriptor(infiniopHandle_t handle,
                                                           infiniopSinhDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y_desc,
                                                           infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Sinh), STATUS_SUCCESS);

    *(_SinhDescriptor_t *) desc_ptr = new _SinhDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopSinh(infiniopSinhDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream) {
    auto _desc = (_SinhDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroySinhDescriptor(infiniopSinhDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_SinhDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}