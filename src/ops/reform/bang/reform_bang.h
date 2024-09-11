#ifndef __BANG_REFORM_H__
#define __BANG_REFORM_H__

#include "../../utils.h"
#include "cnrt.h"
#include "operators.h"

struct ReformBangDescriptor {
    Device device;
};

void reform_bang(Tensor y, Tensor x, void *stream);

#endif// __BANG_REFORM_H__
