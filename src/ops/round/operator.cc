#include "../unary/unary.h"
#include "../utils.h"
#include "ops/round/round.h"
#include "tensor/tensor_descriptor.h"

struct _RoundDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _RoundDescriptor *_RoundDescriptor_t;

__C __export infiniopStatus_t infiniopCreateRoundDescriptor(infiniopHandle_t handle,
                                                            infiniopRoundDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y_desc,
                                                            infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Round), STATUS_SUCCESS);

    *(_RoundDescriptor_t *) desc_ptr = new _RoundDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopRound(infiniopRoundDescriptor_t desc,
                                            void *y,
                                            void const *x,
                                            void *stream) {
    auto _desc = (_RoundDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyRoundDescriptor(infiniopRoundDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_RoundDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}