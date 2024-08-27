#ifndef __CUDA_CAUSAL_SOFTMAX_H__
#define __CUDA_CAUSAL_SOFTMAX_H__

#include "operators.h"

struct CausalSoftmaxCudaDescriptor {
    Device device;
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

infiniopStatus_t cudaCreateCausalSoftmaxDescriptor(infiniopHandle_t handle,
                                                   CausalSoftmaxCudaDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y_desc);

infiniopStatus_t cudaGetCausalSoftmaxWorkspaceSize(CausalSoftmaxCudaDescriptor_t desc, unsigned long int *size);

infiniopStatus_t cudaCausalSoftmax(CausalSoftmaxCudaDescriptor_t desc,
                                   void *workspace,
                                   unsigned long int workspace_size,
                                   void *data,
                                   void *stream);

infiniopStatus_t cudaDestroyCausalSoftmaxDescriptor(CausalSoftmaxCudaDescriptor_t desc);

void causal_softmax_nv_gpu_f16(CausalSoftmaxCudaDescriptor *, Tensor, void *stream);

#endif
