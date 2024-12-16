#ifndef __CNNL_CAUSAL_SOFTMAX_H__
#define __CNNL_CAUSAL_SOFTMAX_H__

#include "../../../devices/bang/bang_handle.h"
#include "cnnl.h"
#include "operators.h"
#include <vector>

struct CausalSoftmaxCnnlDescriptor {
    Device device;
    int device_id;
    std::shared_ptr<Pool<cnnlHandle_t>> pool;
    DT dtype;
    cnnlTensorDescriptor_t yDesc;
    cnnlTensorDescriptor_t maskDesc;
    std::vector<int> dims;
};

typedef struct CausalSoftmaxCnnlDescriptor *CausalSoftmaxCnnlDescriptor_t;

infiniopStatus_t cnnlCreateCausalSoftmaxDescriptor(BangHandle_t handle,
                                                   CausalSoftmaxCnnlDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y_desc);

infiniopStatus_t cnnlGetCausalSoftmaxWorkspaceSize(CausalSoftmaxCnnlDescriptor_t desc, uint64_t *size);

infiniopStatus_t cnnlCausalSoftmax(CausalSoftmaxCnnlDescriptor_t desc,
                                   void *workspace,
                                   unsigned long int workspace_size,
                                   void *data,
                                   void *stream);

infiniopStatus_t cnnlDestroyCausalSoftmaxDescriptor(CausalSoftmaxCnnlDescriptor_t desc);

#endif
