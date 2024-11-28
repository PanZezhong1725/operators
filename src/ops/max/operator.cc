#include "../binary/binary.h"
#include "../utils.h"
#include "ops/max/max.h"
#include "tensor/tensor_descriptor.h"

struct _MaxDescriptor {
    Device device;
    infiniopBinaryDescriptor_t binary_desc;
};

typedef struct _MaxDescriptor *_MaxDescriptor_t;

__C __export infiniopStatus_t infiniopCreateMaxDescriptor(infiniopHandle_t handle,
                                                          infiniopMaxDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c_desc,
                                                          infiniopTensorDescriptor_t a_desc,
                                                          infiniopTensorDescriptor_t b_desc) {
    // binary desc
    infiniopBinaryDescriptor_t binary_desc;
    CHECK_STATUS(infiniopCreateBinaryDescriptor(handle, &binary_desc, c_desc, a_desc, b_desc, BinaryMode::Max), STATUS_SUCCESS);

    *(_MaxDescriptor_t *) desc_ptr = new _MaxDescriptor{
        handle->device,
        binary_desc,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopMax(infiniopMaxDescriptor_t desc,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void *stream) {
    auto _desc = (_MaxDescriptor_t) desc;
    CHECK_STATUS(infiniopBinary(_desc->binary_desc, c, a, b, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyMaxDescriptor(infiniopMaxDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyBinaryDescriptor(((_MaxDescriptor_t) desc)->binary_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}
