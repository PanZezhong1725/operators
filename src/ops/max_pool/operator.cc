#include "../pooling/pooling.h"
#include "../utils.h"
#include "ops/max_pool/max_pool.h"

struct _MaxPoolDescriptor {
    Device device;
    infiniopPoolingDescriptor_t pooling_desc;
    uint64_t workspace_size;
};

typedef struct _MaxPoolDescriptor *_MaxPoolDescriptor_t;

__C __export infiniopStatus_t infiniopCreateMaxPoolDescriptor(infiniopHandle_t handle,
                                                              infiniopMaxPoolDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t y,
                                                              infiniopTensorDescriptor_t x,
                                                              uint64_t const *kernel_shape,
                                                              uint64_t const *pads,
                                                              int64_t const *strides,
                                                              uint64_t n) {
    infiniopPoolingDescriptor_t pooling_desc = new PoolingDescriptor{handle->device};
    CHECK_STATUS(infiniopCreatePoolingDescriptor(handle, &pooling_desc, y, x, kernel_shape, pads, strides, n, 0), STATUS_SUCCESS);
    uint64_t workspace_size = 0;
    CHECK_STATUS(infiniopGetPoolingWorkspaceSize(pooling_desc, &workspace_size), STATUS_SUCCESS);

    *(_MaxPoolDescriptor_t *) desc_ptr = new _MaxPoolDescriptor{
        handle->device,
        pooling_desc,
        workspace_size};

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopGetMaxPoolWorkspaceSize(infiniopMaxPoolDescriptor_t desc, uint64_t *size) {
    *size = ((_MaxPoolDescriptor_t) desc)->workspace_size;
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopMaxPool(infiniopMaxPoolDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void *stream) {
    auto _desc = (_MaxPoolDescriptor_t) desc;
    if (workspace_size < _desc->workspace_size) {
        return STATUS_MEMORY_NOT_ALLOCATED;
    }

    CHECK_STATUS(infiniopPooling(_desc->pooling_desc, workspace, workspace_size, y, x, stream),
                 STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyMaxPoolDescriptor(infiniopMaxPoolDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyPoolingDescriptor(((_MaxPoolDescriptor_t) desc)->pooling_desc), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}
