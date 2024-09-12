#ifndef __MUSA_CAUSAL_SOFTMAX_H__
#define __MUSA_CAUSAL_SOFTMAX_H__

#include "operators.h"
#include "../../../devices/musa/musa_handle.h"

struct CausalSoftmaxMusaDescriptor {
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

typedef struct CausalSoftmaxMusaDescriptor *CausalSoftmaxMusaDescriptor_t;

infiniopStatus_t musaCreateCausalSoftmaxDescriptor(MusaHandle_t handle,
                                                   CausalSoftmaxMusaDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y_desc);

infiniopStatus_t musaGetCausalSoftmaxWorkspaceSize(CausalSoftmaxMusaDescriptor_t desc, unsigned long int *size);

infiniopStatus_t musaCausalSoftmax(CausalSoftmaxMusaDescriptor_t desc,
                                   void *workspace,
                                   unsigned long int workspace_size,
                                   void *data,
                                   void *stream);

infiniopStatus_t musaDestroyCausalSoftmaxDescriptor(CausalSoftmaxMusaDescriptor_t desc);

void causal_softmax_mt_gpu_f16(CausalSoftmaxMusaDescriptor *, Tensor, void *stream);

#endif