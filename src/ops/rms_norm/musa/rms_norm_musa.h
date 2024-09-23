#ifndef __MUSA_RMS_NORM_H__
#define __MUSA_RMS_NORM_H__

#include "operators.h"
#include "../../../devices/musa/musa_handle.h"

struct RMSNormMusaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    unsigned long int n;
    unsigned long int d;
    unsigned long int stride_y;
    unsigned long int stride_x;
    DT w_datatype;
    float epsilon;
    std::shared_ptr<Pool<musa::dnn::Handle>> mudnn_handles_t;
};

typedef struct RMSNormMusaDescriptor *RMSNormMusaDescriptor_t;

infiniopStatus_t musaCreateRMSNormDescriptor(MusaHandle_t handle,
                                             RMSNormMusaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y_desc,
                                             infiniopTensorDescriptor_t x_desc,
                                             infiniopTensorDescriptor_t w_desc,
                                             float epsilon);

infiniopStatus_t musaGetRMSNormWorkspaceSize(RMSNormMusaDescriptor_t desc, unsigned long int *size);

infiniopStatus_t musaRMSNorm(RMSNormMusaDescriptor_t desc,
                                   void *workspace,
                                   unsigned long int workspace_size,
                                   void *y, void *x, void *w,
                                   void *stream);

infiniopStatus_t musaDestroyRMSNormDescriptor(RMSNormMusaDescriptor_t desc);

void rms_norm_mt_gpu_f16(RMSNormMusaDescriptor_t desc, void *y, void *x, void *w, void *stream);
musa::dnn::RMSNorm* createRMSNormOperator(float epsilon);
musa::dnn::Tensor* createMudnnTensor(void const *data, uint64_t batch, uint64_t dim, DT dtype);
musa::dnn::Tensor* createMudnnTensor(void const *data, uint64_t dim, DT dtype);

#endif// __MUSA_RMS_NORM_H__
