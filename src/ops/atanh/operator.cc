#include "../unary/unary.h"
#include "../utils.h"
#include "ops/atanh/atanh.h"
#include "tensor/tensor_descriptor.h"

struct _AtanhDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _AtanhDescriptor *_AtanhDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAtanhDescriptor(infiniopHandle_t handle,
                                                            infiniopAtanhDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y_desc,
                                                            infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Atanh), STATUS_SUCCESS);

    *(_AtanhDescriptor_t *) desc_ptr = new _AtanhDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopAtanh(infiniopAtanhDescriptor_t desc,
                                            void *y,
                                            void const *x,
                                            void *stream) {
    auto _desc = (_AtanhDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyAtanhDescriptor(infiniopAtanhDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_AtanhDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}