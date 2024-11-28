#include "../binary/binary.h"
#include "../utils.h"
#include "ops/mod/mod.h"
#include "tensor/tensor_descriptor.h"

struct _ModDescriptor {
    Device device;
    infiniopBinaryDescriptor_t binary_desc;
};

typedef struct _ModDescriptor *_ModDescriptor_t;

__C __export infiniopStatus_t infiniopCreateModDescriptor(infiniopHandle_t handle,
                                                          infiniopModDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c_desc,
                                                          infiniopTensorDescriptor_t a_desc,
                                                          infiniopTensorDescriptor_t b_desc) {
    // binary desc
    infiniopBinaryDescriptor_t binary_desc;
    CHECK_STATUS(infiniopCreateBinaryDescriptor(handle, &binary_desc, c_desc, a_desc, b_desc, BinaryMode::Mod), STATUS_SUCCESS);

    *(_ModDescriptor_t *) desc_ptr = new _ModDescriptor{
        handle->device,
        binary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopMod(infiniopModDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream) {
    auto _desc = (_ModDescriptor_t) desc;
    CHECK_STATUS(infiniopBinary(_desc->binary_desc, c, a, b, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyModDescriptor(infiniopModDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyBinaryDescriptor(((_ModDescriptor_t) desc)->binary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}
