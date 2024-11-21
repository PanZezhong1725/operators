#include "../binary/binary.h"
#include "../utils.h"
#include "ops/pow/pow.h"
#include "tensor/tensor_descriptor.h"

struct _PowDescriptor {
    Device device;
    infiniopBinaryDescriptor_t binary_desc;
};

typedef struct _PowDescriptor *_PowDescriptor_t;

__C __export infiniopStatus_t infiniopCreatePowDescriptor(infiniopHandle_t handle,
                                                          infiniopPowDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c_desc,
                                                          infiniopTensorDescriptor_t a_desc,
                                                          infiniopTensorDescriptor_t b_desc) {
    // binary desc
    infiniopBinaryDescriptor_t binary_desc;
    CHECK_STATUS(infiniopCreateBinaryDescriptor(handle, &binary_desc, c_desc, a_desc, b_desc, BinaryMode::Pow), STATUS_SUCCESS);

    *(_PowDescriptor_t *) desc_ptr = new _PowDescriptor{
        handle->device,
        binary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopPow(infiniopPowDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream) {
    auto _desc = (_PowDescriptor_t) desc;
    CHECK_STATUS(infiniopBinary(_desc->binary_desc, c, a, b, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyPowDescriptor(infiniopPowDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyBinaryDescriptor(((_PowDescriptor_t) desc)->binary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}
