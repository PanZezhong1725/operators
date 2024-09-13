#include "matmul_musa.h"
#include "../../../devices/musa/common_musa.h"
#include "../../utils.h"

#include <iostream>

infiniopStatus_t musaCreateMatmulDescriptor(MusaHandle_t handle,
                                            MatmulMusaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc) {
    DT dtype = c_desc->dt;
    if (!dtype_eq(dtype, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    infiniopStatus_t *status = new infiniopStatus_t{STATUS_EXECUTION_FAILED};
    auto info = MatmulInfo(c_desc, a_desc, b_desc, status, false);
    if (*status != STATUS_SUCCESS) {
        return *status;
    }

    *desc_ptr = new MatmulMusaDescriptor{
        DevMtGpu,
        dtype,
        handle->device_id,
        info,
        handle->mudnn_handles_t};
    return STATUS_SUCCESS;
}


infiniopStatus_t musaGetMatmulWorkspaceSize(MatmulMusaDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t musaMatmul(MatmulMusaDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *c,
                            float beta,
                            void const *a,
                            void const *b,
                            float alpha,
                            void *stream) {
    if (dtype_eq(desc->dtype, F16)) {
        matmul_musa_f16(desc, c, beta, a, b, alpha, stream);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}

infiniopStatus_t musaDestroyMatmulDescriptor(MatmulMusaDescriptor_t desc) {
    desc->mudnn_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}

void matmul_musa_f16(MatmulMusaDescriptor_t desc, void *c, float beta, void const *a, void const *b, float alpha, void *stream) {
    auto info = desc->info;

    musa::dnn::Tensor *l = createMudnnTensor(a, info.a_matrix, F16);
    musa::dnn::Tensor *r = createMudnnTensor(b, info.b_matrix, F16);
    musa::dnn::Tensor *out = createMudnnTensor(c, info.c_matrix, F16);
    musa::dnn::BatchMatMul *matmul_operator = createMatMulOperator(alpha, beta, true, true);

    use_mudnn(desc->mudnn_handles_t, desc->device_id, (musaStream_t) stream,
              [&](musa::dnn::Handle* handle) {
        size_t size_in_bytes = 0;
        matmul_operator->GetWorkspaceSize(*handle, size_in_bytes, *out, *l, *r);
        matmul_operator->Run(*handle, *out, *l, *r, info.batch,
                             info.m, info.n, info.k, info.a_matrix.ld(),
                             info.b_matrix.ld(), info.c_matrix.ld(),
                             info.a_matrix.stride, info.b_matrix.stride,
                             info.c_matrix.stride, nullptr);
    });
}

musa::dnn::Tensor* createMudnnTensor(void const *data, BlasMatrix matrix, DT dtype) {
    musa::dnn::Tensor* tensor = new musa::dnn::Tensor();

    int64_t* dim = new int64_t;
    if (matrix.ndim == 2) {
        delete dim;
        dim = new int64_t[2]{(int64_t)matrix.rows, (int64_t)matrix.cols};
    }
    else if (matrix.ndim == 3) {
        delete dim;
        dim = new int64_t[3]{(int64_t)matrix.batch, (int64_t)matrix.rows, (int64_t)matrix.cols};
    }

    tensor->SetAddr(data);
    tensor->SetFormat(musa::dnn::Tensor::Format::NCHW);
    tensor->SetNdInfo(matrix.ndim, dim);

    if (dtype_eq(dtype, F16)) {
        tensor->SetType(musa::dnn::Tensor::Type::HALF);
    }
    delete dim;
    return tensor;
}


musa::dnn::BatchMatMul* createMatMulOperator(float alpha, float beta, bool op_a, bool op_b) {
    musa::dnn::BatchMatMul* matmul_operator = new musa::dnn::BatchMatMul();

    matmul_operator->SetComputeMode(musa::dnn::MatMul::ComputeMode::TENSOR);
    matmul_operator->SetAlpha(alpha);
    matmul_operator->SetBeta(beta);

    return matmul_operator;
}