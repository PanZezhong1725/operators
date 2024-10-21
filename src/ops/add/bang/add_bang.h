#ifndef __BANG_ADD_H__
#define __BANG_ADD_H__

#include "../../../devices/bang/bang_handle.h"
#include "../../utils.h"
#include "operators.h"

struct AddBangDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t ndim;
    uint64_t c_data_size;
    uint64_t *c_shape;
    uint64_t *a_strides_d;
    uint64_t *b_strides_d;
};

typedef struct AddBangDescriptor *AddBangDescriptor_t;

infiniopStatus_t bangCreateAddDescriptor(BangHandle_t,
                                         AddBangDescriptor_t *,
                                         infiniopTensorDescriptor_t c,
                                         infiniopTensorDescriptor_t a,
                                         infiniopTensorDescriptor_t b);

infiniopStatus_t bangAdd(AddBangDescriptor_t desc,
                         void *c, void const *a, void const *b,
                         void *stream);

infiniopStatus_t bangDestroyAddDescriptor(AddBangDescriptor_t desc);

#endif
