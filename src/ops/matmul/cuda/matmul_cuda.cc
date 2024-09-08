#include "matmul_cuda.h"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateMatmulDescriptor(CudaHandle_t handle,
                                            MatmulCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc) {
    DT dtype = c_desc->dt;

    if (!dtype_eq(dtype, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    infiniopStatus_t *status = new infiniopStatus_t{STATUS_EXECUTION_FAILED};
    auto info = MatmulInfo(c_desc, a_desc, b_desc, status);
    if (*status != STATUS_SUCCESS) {
        return *status;
    }

    *desc_ptr = new MatmulCudaDescriptor{
        DevNvGpu,
        dtype,
        handle->device_id,
        info,
        handle->cublas_handles_t};
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaMatmul(MatmulCudaDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *c,
                            float beta,
                            void *a,
                            void *b,
                            float alpha,
                            void *stream) {
    if (dtype_eq(desc->dtype, F16)) {
        matmul_cuda_f16(desc, c, beta, a, b, alpha, stream);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}

infiniopStatus_t cudaGetMatmulWorkspaceSize(MatmulCudaDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyMatmulDescriptor(MatmulCudaDescriptor_t desc) {
    desc->cublas_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
