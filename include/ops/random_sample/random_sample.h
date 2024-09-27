#ifndef RANDOM_SAMPLE_H
#define RANDOM_SAMPLE_H

#include "../../export.h"
#include "../../operators.h"

typedef struct RandomSampleDescriptor {
    Device device;
} RandomSampleDescriptor;

typedef RandomSampleDescriptor *infiniopRandomSampleDescriptor_t;

__C __export infiniopStatus_t infiniopCreateRandomSampleDescriptor(infiniopHandle_t handle, infiniopRandomSampleDescriptor_t *desc_ptr, infiniopTensorDescriptor_t probs);

__C __export infiniopStatus_t infiniopGetRandomSampleWorkspaceSize(infiniopRandomSampleDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopRandomSample(infiniopRandomSampleDescriptor_t desc,
                                                   void *workspace,
                                                   uint64_t workspace_size,
                                                   void *result,
                                                   void *probs,
                                                   float random_val,
                                                   float topp,
                                                   int topk,
                                                   float temperature,
                                                   void *stream);

__C __export infiniopStatus_t infiniopDestroyRandomSampleDescriptor(infiniopRandomSampleDescriptor_t desc);


#endif
