#include "random_sample_bang.h"
#include "../../utils.h"

infiniopStatus_t bangCreateRandomSampleDescriptor(BangHandle_t handle,
                                                  RandomSampleBangDescriptor_t *desc_ptr,
                                                  infiniopTensorDescriptor_t probs) {
    if (probs->ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    int voc = probs->shape[0];

    *desc_ptr = new RandomSampleBangDescriptor{
        handle->device,
        handle->device_id,
        probs->dt,
        voc};

    return STATUS_SUCCESS;
}

infiniopStatus_t bangGetRandomSampleWorkspaceSize(RandomSampleBangDescriptor_t desc, unsigned long int *size) {
    *size = desc->voc * (sizeof(int) + sizeof(uint16_t)) + sizeof(uint16_t);
    return STATUS_SUCCESS;
}

infiniopStatus_t bangDestroyRandomSampleDescriptor(RandomSampleBangDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
