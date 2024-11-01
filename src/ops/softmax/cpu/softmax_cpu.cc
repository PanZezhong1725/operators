#include "softmax_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <cmath>

infiniopStatus_t cpuCreateSoftmaxDescriptor(infiniopHandle_t handle,
                                            SoftmaxCpuDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t output_desc) {
    ASSERT_EQ(input_desc->ndim, output_desc->ndim);
    if (!dtype_eq(input_desc->dt, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    int ndim = input_desc->ndim;

    for (int i = 0; i < ndim; i++) {
        if (input_desc->shape[i] != output_desc->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    int *shape = new int[ndim];

    for (int i = 0; i < ndim; i++) {
        shape[i] = static_cast<int>(input_desc->shape[i]);
    }
    *desc_ptr = new SoftmaxCpuDescriptor{
        handle->device,
        input_desc->dt,
        ndim,
        shape};

    return STATUS_SUCCESS;
}


infiniopStatus_t cpuDestroySoftmaxDescriptor(SoftmaxCpuDescriptor_t desc) {
    delete[] desc->shape;
    delete desc;
    return STATUS_SUCCESS;
}
void softmax_cpu_f16(SoftmaxCpuDescriptor_t desc,
                     void const *input, int axis, void *output) {
    auto source = reinterpret_cast<const uint16_t *>(input);
    auto destination = reinterpret_cast<uint16_t *>(output);
    int ndim = desc->ndim;

    auto shape = desc->shape;
    int dimsize = shape[axis];
    int othersize = 1;
    int stride = 1;

    for (int s = ndim - 1; s >= 0; s--) {

        if (s > axis) {
            stride *= shape[s];
        }
        if (s != axis) {
            othersize *= shape[s];
        }
    }
    //假设[I, J, K, S], axis = 1, othersize = IKS
    for (int ind = 0; ind < othersize; ind++) {                 //ind = i(KS) + k(S) + s
        int tid = ind % stride + (ind - ind % stride) * dimsize;//now, tid = i(JKS) + k(S) + s;
        float localM = -__FLT_MAX__;
        for (int j = 0; j < dimsize; j++) {
            int index = tid + j * stride;
            localM = fmax(localM, f16_to_f32(source[index]));
        }
        float localS = 0.0f;
        for (int j = 0; j < dimsize; j++) {
            int index = tid + j * stride;
            localS += std::exp(f16_to_f32(source[index]) - localM);
        }
        for (int j = 0; j < dimsize; j++) {
            int index = tid + j * stride;
            destination[index] = f32_to_f16(std::exp(f16_to_f32(source[index]) - localM) / localS);
        }
    }
}
infiniopStatus_t cpuSoftmax(SoftmaxCpuDescriptor_t desc,
                            void const *input, int axis, void *output,
                            void *stream) {
    if (dtype_eq(desc->dtype, F16)) {
        softmax_cpu_f16(desc,
                        input, axis, output);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
