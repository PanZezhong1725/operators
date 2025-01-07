#include "matmul_kunlun.h"
#include "../../../devices/kunlun/common_kunlun.h"
#include "../../utils.h"

template<typename T>
infiniopStatus_t matmul_kunlun(MatmulKunlunDescriptor_t desc,
                               void *c,
                               float beta,
                               void const *a,
                               void const *b,
                               float alpha,
                               void *stream) {
    auto info = desc->info;

    if (info.is_transed) {
        std::swap(a, b);
    }

    auto transA = info.a_matrix.col_stride == 1 ? false : true;
    auto transB = info.b_matrix.col_stride == 1 ? false : true;
    // int64_t strideA = transA ? info.a_matrix.col_stride * info.a_matrix.cols
    //                          : info.a_matrix.row_stride * info.a_matrix.rows;
    // int64_t strideB = transB ? info.b_matrix.col_stride * info.b_matrix.cols
    //                          : info.b_matrix.row_stride * info.b_matrix.rows;
    // int64_t strideC = info.batch == 1
    //                       ? info.c_matrix.row_stride * info.c_matrix.rows
    //                       : info.c_matrix.stride;
    use_xdnn(desc->xdnn_handles_t,
             desc->device_id,
             (XPUStream) stream,
             [&](xdnnHandle_t handle) {
                 //  xdnn::fc_batched<T, T, T, int16_t>(handle, info.batch, transA,transB,info.m,info.n,info.k,alpha,(T *) a,strideA,(T *) b,strideB,beta,(T *) c,strideC,nullptr,nullptr);
                 for (int i = 0; i < info.batch; i++) {
                     checkKUNLUNError((
                         xdnn::fc_fusion<T, T, T, int16_t>(
                             handle,
                             (T *) ((char *) a + i * info.a_matrix.stride * (desc->dtype).size),
                             (T *) ((char *) b + i * info.b_matrix.stride * (desc->dtype).size),
                             (T *) ((char *) c + i * info.c_matrix.stride * (desc->dtype).size),
                             info.m,
                             info.n,
                             info.k,
                             transA,
                             transB,
                             nullptr,
                             nullptr,
                             nullptr,
                             info.a_matrix.ld(),
                             info.b_matrix.ld(),
                             info.c_matrix.ld(),
                             alpha,
                             beta,
                             nullptr,
                             xdnn::Activation_t::LINEAR,
                             nullptr)));
                 }
             });
    return STATUS_SUCCESS;
}


infiniopStatus_t kunlunCreateMatmulDescriptor(KunlunHandle_t handle,
                                              MatmulKunlunDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t c_desc,
                                              float alpha,
                                              infiniopTensorDescriptor_t a_desc,
                                              infiniopTensorDescriptor_t b_desc,
                                              float beta) {
    DT dtype = c_desc->dt;

    if (dtype != F16 && dtype != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    infiniopStatus_t *status = new infiniopStatus_t{STATUS_EXECUTION_FAILED};
    auto info = MatmulInfo(c_desc, a_desc, b_desc, status, false);
    if (*status != STATUS_SUCCESS) {
        return *status;
    }

    *desc_ptr = new MatmulKunlunDescriptor{
        DevKunlunXpu,
        dtype,
        handle->device_id,
        info,
        alpha,
        beta,
        handle->xdnn_handles_t};
    return STATUS_SUCCESS;
}

infiniopStatus_t kunlunMatmul(MatmulKunlunDescriptor_t desc,
                              void *workspace,
                              uint64_t workspace_size,
                              void *c,
                              void const *a,
                              void const *b,
                              void *stream) {
    if (desc->dtype == F16) {
        return matmul_kunlun<float16>(desc, c, desc->beta, a, b, desc->alpha, stream);
    }
    if (desc->dtype == F32) {
        return matmul_kunlun<float>(desc, c, desc->beta, a, b, desc->alpha, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}


infiniopStatus_t kunlunGetMatmulWorkspaceSize(MatmulKunlunDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t kunlunDestroyMatmulDescriptor(MatmulKunlunDescriptor_t desc) {
    desc->xdnn_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
