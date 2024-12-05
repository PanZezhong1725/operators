#include "../unary/unary.h"
#include "../utils.h"
#include "ops/floor/floor.h"
#include "tensor/tensor_descriptor.h"

struct _FloorDescriptor {
    Device device;
    infiniopUnaryDescriptor_t unary_desc;
};

typedef struct _FloorDescriptor *_FloorDescriptor_t;

__C __export infiniopStatus_t infiniopCreateFloorDescriptor(infiniopHandle_t handle,
                                                            infiniopFloorDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y_desc,
                                                            infiniopTensorDescriptor_t x_desc) {
    // unary desc
    infiniopUnaryDescriptor_t unary_desc;
    CHECK_STATUS(infiniopCreateUnaryDescriptor(handle, &unary_desc, y_desc, x_desc, UnaryMode::Floor), STATUS_SUCCESS);

    *(_FloorDescriptor_t *) desc_ptr = new _FloorDescriptor{
        handle->device,
        unary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopFloor(infiniopFloorDescriptor_t desc,
                                            void *y,
                                            void const *x,
                                            void *stream) {
    auto _desc = (_FloorDescriptor_t) desc;
    CHECK_STATUS(infiniopUnary(_desc->unary_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyFloorDescriptor(infiniopFloorDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyUnaryDescriptor(((_FloorDescriptor_t) desc)->unary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}