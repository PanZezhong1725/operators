#include "../binary/binary.h"
#include "../utils.h"
#include "ops/div/div.h"
#include "tensor/tensor_descriptor.h"

struct _DivDescriptor {
    Device device;
    infiniopBinaryDescriptor_t binary_desc;
};

typedef struct _DivDescriptor *_DivDescriptor_t;

__C __export infiniopStatus_t infiniopCreateDivDescriptor(infiniopHandle_t handle,
                                                          infiniopDivDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c_desc,
                                                          infiniopTensorDescriptor_t a_desc,
                                                          infiniopTensorDescriptor_t b_desc) {
    // binary desc
    infiniopBinaryDescriptor_t binary_desc;
    CHECK_STATUS(infiniopCreateBinaryDescriptor(handle, &binary_desc, c_desc, a_desc, b_desc, BinaryMode::Divide), STATUS_SUCCESS);

    *(_DivDescriptor_t *) desc_ptr = new _DivDescriptor{
        handle->device,
        binary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDiv(infiniopDivDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream) {
    auto _desc = (_DivDescriptor_t) desc;
    CHECK_STATUS(infiniopBinary(_desc->binary_desc, c, a, b, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyDivDescriptor(infiniopDivDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyBinaryDescriptor(((_DivDescriptor_t) desc)->binary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}
