#ifndef __CPU_RANDOM_SAMPLE_H__
#define __CPU_RANDOM_SAMPLE_H__

#include "operators.h"
struct RandomSampleCpuDescriptor {
    Device device;
    DT dtype;
    int voc;
};

typedef struct RandomSampleCpuDescriptor *RandomSampleCpuDescriptor_t;

infiniopStatus_t cpuCreateRandomSampleDescriptor(infiniopHandle_t,
                                                 RandomSampleCpuDescriptor_t *,
                                                 infiniopTensorDescriptor_t probs);

infiniopStatus_t cpuGetRandomSampleWorkspaceSize(RandomSampleCpuDescriptor_t desc, uint64_t *size);

infiniopStatus_t cpuRandomSample(RandomSampleCpuDescriptor_t desc,
                                 void *workspace,
                                 uint64_t workspace_size,
                                 void *result,
                                 void *probs,
                                 float random_val,
                                 float topp,
                                 int topk,
                                 float temperature,
                                 void *stream);

infiniopStatus_t cpuDestroyRandomSampleDescriptor(RandomSampleCpuDescriptor_t desc);

#endif
