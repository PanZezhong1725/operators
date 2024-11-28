#include "../binary/binary.h"
#include "../utils.h"
#include "ops/add/add.h"
#include "tensor/tensor_descriptor.h"

struct _AddDescriptor {
    Device device;
    infiniopBinaryDescriptor_t binary_desc;
};

typedef struct _AddDescriptor *_AddDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAddDescriptor(infiniopHandle_t handle,
                                                          infiniopAddDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c_desc,
                                                          infiniopTensorDescriptor_t a_desc,
                                                          infiniopTensorDescriptor_t b_desc) {
    // binary desc
    infiniopBinaryDescriptor_t binary_desc;
    CHECK_STATUS(infiniopCreateBinaryDescriptor(handle, &binary_desc, c_desc, a_desc, b_desc, BinaryMode::Add), STATUS_SUCCESS);

    *(_AddDescriptor_t *) desc_ptr = new _AddDescriptor{
        handle->device,
        binary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopAdd(infiniopAddDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream) {
    auto _desc = (_AddDescriptor_t) desc;
    CHECK_STATUS(infiniopBinary(_desc->binary_desc, c, a, b, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyAddDescriptor(infiniopAddDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyBinaryDescriptor(((_AddDescriptor_t) desc)->binary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}
