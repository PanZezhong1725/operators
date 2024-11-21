#include "../binary/binary.h"
#include "../utils.h"
#include "ops/sub/sub.h"
#include "tensor/tensor_descriptor.h"

struct _SubDescriptor {
    Device device;
    infiniopBinaryDescriptor_t binary_desc;
};

typedef struct _SubDescriptor *_SubDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSubDescriptor(infiniopHandle_t handle,
                                                          infiniopSubDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c_desc,
                                                          infiniopTensorDescriptor_t a_desc,
                                                          infiniopTensorDescriptor_t b_desc) {
    // binary desc
    infiniopBinaryDescriptor_t binary_desc;
    CHECK_STATUS(infiniopCreateBinaryDescriptor(handle, &binary_desc, c_desc, a_desc, b_desc, BinaryMode::Subtract), STATUS_SUCCESS);

    *(_SubDescriptor_t *) desc_ptr = new _SubDescriptor{
        handle->device,
        binary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopSub(infiniopSubDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream) {
    auto _desc = (_SubDescriptor_t) desc;
    CHECK_STATUS(infiniopBinary(_desc->binary_desc, c, a, b, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroySubDescriptor(infiniopSubDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyBinaryDescriptor(((_SubDescriptor_t) desc)->binary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}
