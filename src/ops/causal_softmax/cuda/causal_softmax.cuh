#ifndef __CUDA_CAUSAL_SOFTMAX_H__
#define __CUDA_CAUSAL_SOFTMAX_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"

struct CausalSoftmaxCudaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    unsigned long int batch_size;
    unsigned long int stride_b;
    unsigned long int seq_len;
    unsigned long int stride_i;
    unsigned long int total_seq_len;
    unsigned long int stride_j;
    unsigned int max_items_per_thread;
};

typedef struct CausalSoftmaxCudaDescriptor *CausalSoftmaxCudaDescriptor_t;

infiniopStatus_t cudaCreateCausalSoftmaxDescriptor(CudaHandle_t handle,
                                                   CausalSoftmaxCudaDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y_desc);

infiniopStatus_t cudaGetCausalSoftmaxWorkspaceSize(CausalSoftmaxCudaDescriptor_t desc, uint64_t *size);

infiniopStatus_t cudaCausalSoftmax(CausalSoftmaxCudaDescriptor_t desc,
                                   void *workspace,
                                   unsigned long int workspace_size,
                                   void *data,
                                   void *stream);

infiniopStatus_t cudaDestroyCausalSoftmaxDescriptor(CausalSoftmaxCudaDescriptor_t desc);

#endif
