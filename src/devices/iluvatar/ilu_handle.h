#ifndef ILU_HANDLE_H
#define ILU_HANDLE_H

#include "../../../include/device.h"
#include "../../../include/status.h"
#include "../pool.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <memory>

struct IluContext {
    Device device;
    int device_id;
    std::shared_ptr<Pool<cublasHandle_t>> cublas_handles_t;
    std::shared_ptr<Pool<cudnnHandle_t>> cudnn_handles_t;
    cudaDeviceProp prop;
    int compute_capability_major;
    int compute_capability_minor;
};
typedef struct IluContext *IluHandle_t;

infiniopStatus_t createIluHandle(IluHandle_t *handle_ptr, int device_id);

infiniopStatus_t deleteIluHandle(IluHandle_t handle_ptr);

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

template<typename T>
cudnnStatus_t use_cudnn(std::shared_ptr<Pool<cudnnHandle_t>> cudnn_handles_t, int device_id, cudaStream_t stream, T const &f) {
    auto handle = cudnn_handles_t->pop();
    if (!handle) {
        cudaSetDevice(device_id);
        cudnnCreate(&(*handle));
    }
    cudnnSetStream(*handle, stream);
    cudnnStatus_t status = f(*handle);
    cudnn_handles_t->push(std::move(*handle));
    return status;
}

#endif
