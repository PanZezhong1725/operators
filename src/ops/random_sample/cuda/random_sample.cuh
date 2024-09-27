#ifndef __CUDA_RANDOM_SAMPLE_H__
#define __CUDA_RANDOM_SAMPLE_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"

struct RandomSampleCudaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    int voc;
};

typedef struct RandomSampleCudaDescriptor *RandomSampleCudaDescriptor_t;

infiniopStatus_t cudaCreateRandomSampleDescriptor(CudaHandle_t handle,
                                                  RandomSampleCudaDescriptor_t *desc_ptr,
                                                  infiniopTensorDescriptor_t probs);

infiniopStatus_t cudaGetRandomSampleWorkspaceSize(RandomSampleCudaDescriptor_t desc, unsigned long int *size);

infiniopStatus_t cudaRandomSample(RandomSampleCudaDescriptor_t desc,
                                  void *workspace,
                                  uint64_t workspace_size,
                                  void *result,
                                  void *probs,
                                  float random_val,
                                  float topp,
                                  int topk,
                                  float temperature,
                                  void *stream);

infiniopStatus_t cudaDestroyRandomSampleDescriptor(RandomSampleCudaDescriptor_t desc);


#endif
