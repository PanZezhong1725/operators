#ifndef __CUDA_REARRANGE_H__
#define __CUDA_REARRANGE_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"

struct RearrangeCudaDescriptor {
    Device device;
    int device_id;
    unsigned long int rsa;
    unsigned long int rsb;
    unsigned long int csa;
    unsigned long int csb;
    unsigned long int r, c, b;
    unsigned long int bytes_per_thread;
};

typedef struct RearrangeCudaDescriptor *RearrangeCudaDescriptor_t;

infiniopStatus_t cudaCreateRearrangeDescriptor(CudaHandle_t handle,
                                               RearrangeCudaDescriptor_t *desc_ptr,
                                               infiniopTensorDescriptor_t dst,
                                               infiniopTensorDescriptor_t src);

infiniopStatus_t cudaRearrange(RearrangeCudaDescriptor_t desc,
                               void *dst,
                               void *src,
                               void *stream);

infiniopStatus_t cudaDestroyRearrangeDescriptor(RearrangeCudaDescriptor_t desc);

void rearrange_nv_gpu(RearrangeCudaDescriptor *, void *y, void const *x, void *stream);
#endif// __CUDA_REARRANGE_H__
