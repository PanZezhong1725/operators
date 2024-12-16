#ifndef __BANG_CAUSAL_SOFTMAX_H__
#define __BANG_CAUSAL_SOFTMAX_H__

#include "../../../devices/bang/bang_handle.h"
#include "../../utils.h"
#include "operators.h"

struct CausalSoftmaxBangDescriptor {
    Device device;
    int device_id;
    DT dtype;
    int ndim;
    int *stride;
    int *shape;
    int n;
};

typedef struct CausalSoftmaxBangDescriptor *CausalSoftmaxBangDescriptor_t;

infiniopStatus_t bangCreateCausalSoftmaxDescriptor(BangHandle_t handle,
                                                   CausalSoftmaxBangDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y_desc);

infiniopStatus_t bangGetCausalSoftmaxWorkspaceSize(CausalSoftmaxBangDescriptor_t desc, uint64_t *size);

infiniopStatus_t bangCausalSoftmax(CausalSoftmaxBangDescriptor_t desc,
                                   void *workspace,
                                   unsigned long int workspace_size,
                                   void *data,
                                   void *stream);

infiniopStatus_t bangDestroyCausalSoftmaxDescriptor(CausalSoftmaxBangDescriptor_t desc);


#endif
