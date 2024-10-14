#include "causal_softmax_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../utils.h"
#include <algorithm>

infiniopStatus_t cpuCreateCausalSoftmaxDescriptor(infiniopHandle_t,
                                                  CausalSoftmaxCpuDescriptor_t *desc_ptr,
                                                  infiniopTensorDescriptor_t y) {
    uint64_t ndim = y->ndim;
    if (ndim != 2 && ndim != 3) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!dtype_eq(y->dt, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    uint64_t total_seq_len = y->shape[ndim - 1];
    uint64_t seq_len = y->shape[ndim - 2];
    uint64_t batch_size = 1;
    uint64_t stride_j = y->strides[ndim - 1];
    uint64_t stride_i = y->strides[ndim - 2];
    uint64_t stride_b = 0;
    if (ndim == 3)
        stride_b = y->strides[ndim - 3];
    for (size_t i = 0; i < ndim - 2; i++) {
        batch_size *= y->shape[i];
    }

    *desc_ptr = new CausalSoftmaxCpuDescriptor{
        DevCpu,
        y->dt,
        batch_size,
        stride_b,
        seq_len,
        stride_i,
        total_seq_len,
        stride_j};

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGetCausalSoftmaxWorkspaceSize(CausalSoftmaxCpuDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyCausalSoftmaxDescriptor(CausalSoftmaxCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}


void causal_softmax_cpu_f16(CausalSoftmaxCpuDescriptor_t desc, void* y) {
    uint64_t total_seq_len = desc->total_seq_len;
    uint64_t seq_len = desc->seq_len;
    uint64_t batch_size = desc->batch_size;
    uint64_t stride_j = desc->stride_j;
    uint64_t stride_i = desc->stride_i;
    uint64_t stride_b = desc->stride_b;
    auto y_ptr = reinterpret_cast<uint16_t *>(y);
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < seq_len; i++) {
            uint64_t offset = b * stride_b + i * stride_i;
            float max_val = f16_to_f32(y_ptr[offset]);
            for (size_t j = 1; j < total_seq_len; j++) {
                if (j <= total_seq_len - seq_len + i) {
                    max_val = std::max(max_val, f16_to_f32(y_ptr[offset + j * stride_j]));
                } else {
                    y_ptr[offset + j * stride_j] = 0;
                }
            }
            float sum = 0.;
            for (size_t j = 0; j <= total_seq_len - seq_len + i; j++) {
                float new_val = std::exp(f16_to_f32(y_ptr[offset + j * stride_j]) - max_val);
                sum += new_val;
            }
            for (size_t j = 0; j <= total_seq_len - seq_len + i; j++) {
                float new_val = std::exp(f16_to_f32(y_ptr[offset + j * stride_j]) - max_val) / sum;
                y_ptr[offset + j * stride_j] = f32_to_f16(new_val);
            }
        }
    }
}

infiniopStatus_t cpuCausalSoftmax(CausalSoftmaxCpuDescriptor_t desc,
                                  void *workspace,
                                  uint64_t workspace_size,
                                  void *data,
                                  void *stream) {
    if(dtype_eq(desc->dtype, F16)){
        causal_softmax_cpu_f16(desc, data);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
