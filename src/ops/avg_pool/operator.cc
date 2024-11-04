#include "../utils.h"
#include "ops/avg_pool/avg_pool.h"
#include "ops/pooling/pooling.h"

struct _AvgPoolDescriptor {
    Device device;
    infiniopPoolingDescriptor_t pooling_desc;
    uint64_t workspace_size;
};

typedef struct _AvgPoolDescriptor *_AvgPoolDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAvgPoolDescriptor(infiniopHandle_t handle,
                                                              infiniopAvgPoolDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t y,
                                                              infiniopTensorDescriptor_t x,
                                                              void const *kernel_shape,
                                                              void const *pads,
                                                              void const *strides,
                                                              uint64_t n) {
    infiniopPoolingDescriptor_t pooling_desc = new PoolingDescriptor{handle->device};
    CHECK_STATUS(infiniopCreatePoolingDescriptor(handle, &pooling_desc, y, x, kernel_shape, pads, strides, n, 1), STATUS_SUCCESS);
    uint64_t workspace_size = 0;
    CHECK_STATUS(infiniopGetPoolingWorkspaceSize(pooling_desc, &workspace_size), STATUS_SUCCESS);

    *(_AvgPoolDescriptor_t *) desc_ptr = new _AvgPoolDescriptor{
        handle->device,
        pooling_desc,
        workspace_size};

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopGetAvgPoolWorkspaceSize(infiniopAvgPoolDescriptor_t desc, uint64_t *size) {
    *size = ((_AvgPoolDescriptor_t) desc)->workspace_size;
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopAvgPool(infiniopAvgPoolDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void *stream) {
    auto _desc = (_AvgPoolDescriptor_t) desc;
    if (workspace_size < _desc->workspace_size) {
        return STATUS_MEMORY_NOT_ALLOCATED;
    }

    CHECK_STATUS(infiniopPooling(_desc->pooling_desc, workspace, workspace_size, y, x, stream),
                 STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyAvgPoolDescriptor(infiniopAvgPoolDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyPoolingDescriptor(((_AvgPoolDescriptor_t) desc)->pooling_desc), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}
