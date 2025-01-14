#ifndef __KUNLUN_MATMUL_H__
#define __KUNLUN_MATMUL_H__

#include "../../../devices/kunlun/kunlun_handle.h"
#include "../../utils.h"
#include "../blas.h"
#include "operators.h"

typedef struct MatmulKunlunDescriptor {
    Device device;
    DT dtype;
    int device_id;
    MatmulInfo info;
    float alpha;
    float beta;
    std::shared_ptr<Pool<xdnnHandle_t>> xdnn_handles_t;
} MatmulKunlunDescriptor;

typedef struct MatmulKunlunDescriptor *MatmulKunlunDescriptor_t;

infiniopStatus_t kunlunCreateMatmulDescriptor(KunlunHandle_t handle,
                                              MatmulKunlunDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t c_desc,
                                              float alpha,
                                              infiniopTensorDescriptor_t a_desc,
                                              infiniopTensorDescriptor_t b_desc,
                                              float beta);

infiniopStatus_t kunlunGetMatmulWorkspaceSize(MatmulKunlunDescriptor_t desc, uint64_t *size);

infiniopStatus_t kunlunMatmul(MatmulKunlunDescriptor_t desc,
                              void *workspace,
                              uint64_t workspace_size,
                              void *c,
                              void const *a,
                              void const *b,
                              void *stream);

infiniopStatus_t kunlunDestroyMatmulDescriptor(MatmulKunlunDescriptor_t desc);

#endif
