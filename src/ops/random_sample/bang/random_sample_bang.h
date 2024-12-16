#ifndef __BANG_RANDOM_SAMPLE_H__
#define __BANG_RANDOM_SAMPLE_H__

#include "../../../devices/bang/bang_handle.h"
#include "../../utils.h"
#include "operators.h"

struct RandomSampleBangDescriptor {
    Device device;
    int device_id;
    DT dtype;
    int voc;
    DT rDtype;
    int rLength;
};

typedef struct RandomSampleBangDescriptor *RandomSampleBangDescriptor_t;

infiniopStatus_t bangCreateRandomSampleDescriptor(BangHandle_t handle,
                                                  RandomSampleBangDescriptor_t *desc_ptr, infiniopTensorDescriptor_t result,
                                                  infiniopTensorDescriptor_t probs);

infiniopStatus_t bangGetRandomSampleWorkspaceSize(RandomSampleBangDescriptor_t desc, uint64_t *size);

infiniopStatus_t bangRandomSample(RandomSampleBangDescriptor_t desc,
                                  void *workspace,
                                  unsigned long int workspace_size,
                                  void *result,
                                  void const *probs,
                                  float random_val,
                                  float topp,
                                  int topk,
                                  float temperature,
                                  void *stream);

infiniopStatus_t bangDestroyRandomSampleDescriptor(RandomSampleBangDescriptor_t desc);


#endif
