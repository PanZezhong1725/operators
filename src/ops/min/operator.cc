#include "../binary/binary.h"
#include "../utils.h"
#include "ops/min/min.h"
#include "tensor/tensor_descriptor.h"

struct _MinDescriptor {
    Device device;
    infiniopBinaryDescriptor_t binary_desc;
};

typedef struct _MinDescriptor *_MinDescriptor_t;

__C __export infiniopStatus_t infiniopCreateMinDescriptor(infiniopHandle_t handle,
                                                          infiniopMinDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c_desc,
                                                          infiniopTensorDescriptor_t a_desc,
                                                          infiniopTensorDescriptor_t b_desc) {
    // binary desc
    infiniopBinaryDescriptor_t binary_desc;
    CHECK_STATUS(infiniopCreateBinaryDescriptor(handle, &binary_desc, c_desc, a_desc, b_desc, BinaryMode::Min), STATUS_SUCCESS);

    *(_MinDescriptor_t *) desc_ptr = new _MinDescriptor{
        handle->device,
        binary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopMin(infiniopMinDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream) {
    auto _desc = (_MinDescriptor_t) desc;
    CHECK_STATUS(infiniopBinary(_desc->binary_desc, c, a, b, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyMinDescriptor(infiniopMinDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyBinaryDescriptor(((_MinDescriptor_t) desc)->binary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}
