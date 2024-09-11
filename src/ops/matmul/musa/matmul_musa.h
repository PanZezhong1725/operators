#ifndef __MUSA_MATMUL_H__
#define __MUSA_MATMUL_H__

#include <memory>
#include <musa.h>
#include <musa_runtime.h>
#include <mudnn.h>
#include <mudnn_base.h>
#include "../blas.h"
#include "operators.h"
#include "../../../devices/musa/musa_handle.h"

typedef struct MatmulMusaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    MatmulInfo info;
    std::shared_ptr<Pool<musa::dnn::Handle>> mudnn_handles_t;
} MatmulMusaDescriptor;

typedef struct MatmulMusaDescriptor *MatmulMusaDescriptor_t;

infiniopStatus_t musaCreateMatmulDescriptor(MusaHandle_t handle,
                                            MatmulMusaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc);

infiniopStatus_t musaGetMatmulWorkspaceSize(MatmulMusaDescriptor_t desc, uint64_t *size);

infiniopStatus_t musaMatmul(MatmulMusaDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *c,
                            float beta,
                            void const *a,
                            void const *b,
                            float alpha,
                            void *stream);

infiniopStatus_t musaDestroyMatmulDescriptor(MatmulMusaDescriptor_t desc);

void matmul_musa_f16(MatmulMusaDescriptor_t desc, void *c, float beta, void const *a, void const *b, float alpha, void *stream);

musa::dnn::Tensor* createMudnnTensor(void const *data, BlasMatrix matrix, DT dtype);
musa::dnn::BatchMatMul* createMatMulOperator(float alpha, float beta, bool op_a, bool op_b);

#endif // __MUSA_MATMUL_H__