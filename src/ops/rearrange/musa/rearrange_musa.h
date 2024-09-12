#ifndef __MUSA_REARRANGE_H__
#define __MUSA_REARRANGE_H__

#include "operators.h"
#include "../../../devices/musa/musa_handle.h"

struct RearrangeMusaDescriptor {
    Device device;
    int device_id;
    unsigned long int rsa;
    unsigned long int rsb;
    unsigned long int csa;
    unsigned long int csb;
    unsigned long int r, c, b;
    unsigned long int bytes_per_thread;
};

typedef struct RearrangeMusaDescriptor *RearrangeMusaDescriptor_t;

infiniopStatus_t musaCreateRearrangeDescriptor(MusaHandle_t handle,
                                               RearrangeMusaDescriptor_t *desc_ptr,
                                               infiniopTensorDescriptor_t dst,
                                               infiniopTensorDescriptor_t src);

infiniopStatus_t musaRearrange(RearrangeMusaDescriptor_t desc,
                               void *dst,
                               void *src,
                               void *stream);

infiniopStatus_t musaDestroyRearrangeDescriptor(RearrangeMusaDescriptor_t desc);

void rearrange_mt_gpu(RearrangeMusaDescriptor *, void *y, void *x, void *stream);
#endif // __MUSA_REARRANGE_H__
