#ifndef REFORM_H
#define REFORM_H

/* @deprecated This operator is renamed, and the whole file will be removed. */

#include "../../export.h"
#include "../../operators.h"
typedef struct ReformDescriptor ReformDescriptor;

__C __export ReformDescriptor *createReformDescriptor(Device, void *config);
__C __export void destroyReformDescriptor(ReformDescriptor *descriptor);
__C __export void reform(ReformDescriptor *descriptor, Tensor y, Tensor x, void *stream);

#endif
