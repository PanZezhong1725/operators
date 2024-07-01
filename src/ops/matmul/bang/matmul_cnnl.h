#ifndef __CNNL_MATMUL_H__
#define __CNNL_MATMUL_H__

#include "../../../operators.h"
#include "cnnl.h"
#include "cnnl_extra.h"

struct MatmulBangDescriptor {
    Device device;
    MatmulBangDescriptor(Device device);
};

void matmul_cnnl_f16(Tensor c, float beta, Tensor a, Tensor b, float alpha, void *stream);

#endif// __CNNL_MATMUL_H__
