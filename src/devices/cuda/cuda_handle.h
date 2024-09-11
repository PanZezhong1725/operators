#ifndef CUDA_HANDLE_H
#define CUDA_HANDLE_H

#include "../pool.h"
#include "device.h"
#include "ops/matmul/matmul.h"
#include "status.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>

struct CudaContext {
    Device device;
    int device_id;
    std::shared_ptr<Pool<cublasHandle_t>> cublas_handles_t;
};
typedef struct CudaContext *CudaHandle_t;

infiniopStatus_t createCudaHandle(CudaHandle_t *handle_ptr, int device_id);

infiniopStatus_t deleteCudaHandle(CudaHandle_t handle_ptr);

template<typename T>
void use_cublas(std::shared_ptr<Pool<cublasHandle_t>> cublas_handles_t, int device_id, cudaStream_t stream, T const &f) {
    auto handle = cublas_handles_t->pop();
    if (!handle) {
        cudaSetDevice(device_id);
        cublasCreate(&(*handle));
    }
    cublasSetStream(*handle, (cudaStream_t) stream);
    f(*handle);
    cublas_handles_t->push(std::move(*handle));
}

#endif
