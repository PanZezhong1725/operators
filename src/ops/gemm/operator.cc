#include "../utils.h"
#include "ops/expand/expand.h"
#include "ops/gemm/gemm.h"
#include "ops/matmul/matmul.h"
#include "tensor/tensor_descriptor.h"

struct _GEMMDescriptor {
    Device device;
    infiniopMatmulDescriptor_t matmul_desc;
    infiniopExpandDescriptor_t expand_desc;
    uint64_t workspace_size;
};

typedef struct _GEMMDescriptor *_GEMMDescriptor_t;

__C __export infiniopStatus_t infiniopCreateGEMMDescriptor(infiniopHandle_t handle,
                                                           infiniopGEMMDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y_desc,
                                                           infiniopTensorDescriptor_t a_desc,
                                                           infiniopTensorDescriptor_t b_desc,
                                                           infiniopTensorDescriptor_t c_desc,
                                                           float alpha,
                                                           float beta,
                                                           char transA,
                                                           char transB) {
    // transpose a and b if needed
    a_desc = transA ? permute(a_desc, {1, 0}) : a_desc;
    b_desc = transB ? permute(b_desc, {1, 0}) : b_desc;

    // expand desc
    infiniopExpandDescriptor_t expand_desc = nullptr;

    // c is optional, set beta to 0 when c is not provided
    if (!c_desc || c_desc->ndim == 0 || c_desc->shape == nullptr || c_desc->shape[0] == 0) {
        beta = 0;
    } else {
        expand_desc = new ExpandDescriptor{handle->device};
        CHECK_STATUS(infiniopCreateExpandDescriptor(handle, &expand_desc, y_desc, c_desc), STATUS_SUCCESS);
    }

    // matmul desc
    infiniopMatmulDescriptor_t matmul_desc = new MatmulDescriptor{handle->device};
    CHECK_STATUS(infiniopCreateMatmulDescriptor(handle, &matmul_desc, y_desc, alpha, a_desc, b_desc, beta), STATUS_SUCCESS);
    uint64_t workspace_size = 0;
    CHECK_STATUS(infiniopGetMatmulWorkspaceSize(matmul_desc, &workspace_size), STATUS_SUCCESS);

    *(_GEMMDescriptor_t *) desc_ptr = new _GEMMDescriptor{
        handle->device,
        matmul_desc,
        expand_desc,
        workspace_size,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopGetGEMMWorkspaceSize(infiniopGEMMDescriptor_t desc, uint64_t *size) {
    *size = ((_GEMMDescriptor_t) desc)->workspace_size;
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopGEMM(infiniopGEMMDescriptor_t desc,
                                           void *workspace,
                                           uint64_t workspace_size,
                                           void *y,
                                           void const *a,
                                           void const *b,
                                           void const *c,
                                           void *stream) {
    auto _desc = (_GEMMDescriptor_t) desc;
    if (workspace_size < _desc->workspace_size) {
        return STATUS_MEMORY_NOT_ALLOCATED;
    }

    if (_desc->expand_desc != nullptr) {
        CHECK_STATUS(infiniopExpand(_desc->expand_desc,
                                    y, c, stream),
                     STATUS_SUCCESS);
    }

    CHECK_STATUS(infiniopMatmul(_desc->matmul_desc,
                                workspace,
                                workspace_size,
                                y, a, b, stream),
                 STATUS_SUCCESS);

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyGEMMDescriptor(infiniopGEMMDescriptor_t desc) {
    if (((_GEMMDescriptor_t) desc)->expand_desc) {
        CHECK_STATUS(infiniopDestroyExpandDescriptor(((_GEMMDescriptor_t) desc)->expand_desc), STATUS_SUCCESS);
    }
    CHECK_STATUS(infiniopDestroyMatmulDescriptor(((_GEMMDescriptor_t) desc)->matmul_desc), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}
