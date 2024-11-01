#ifndef __CNNL_SOFTMAX_H__
#define __CNNL_SOFTMAX_H__
#include "../../../devices/bang/bang_handle.h"
#include "cnnl.h"
#include "cnnl_extra.h"
#include "operators.h"

struct SoftmaxCnnlDescriptor {
    Device device;
    int device_id;
    DT dtype;
    std::shared_ptr<Pool<cnnlHandle_t>> cnnl_handles;
    int ndim;
    int *shape;
};
typedef struct SoftmaxCnnlDescriptor *SoftmaxCnnlDescriptor_t;

infiniopStatus_t cnnlCreateSoftmaxDescriptor(BangHandle_t handle,
                                             SoftmaxCnnlDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t input_desc,
                                             infiniopTensorDescriptor_t output_desc);


infiniopStatus_t cnnlSoftmax(SoftmaxCnnlDescriptor_t desc, void const *input, int axis, void *output, void *stream);

infiniopStatus_t cnnlDestroySoftmaxDescriptor(SoftmaxCnnlDescriptor_t desc);


#endif// __CNNL_SOFTMAX_H__
