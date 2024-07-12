#include "softmax_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <algorithm>
#include <iostream>

void softmax_cpu_f16(Tensor y) {
    ASSERT(y.layout->ndim >= 2);
    uint64_t total_seq_len = y.layout->shape[y.layout->ndim - 1];
    uint64_t seq_len = y.layout->shape[y.layout->ndim - 2];
    uint64_t batch_size = 1;
    for (size_t i = 0; i < y.layout->ndim - 2; i++) {
        batch_size *= y.layout->shape[i];
    }
    auto y_ptr = reinterpret_cast<uint16_t *>(y.data);
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < seq_len; i++) {
            uint64_t offset = b * total_seq_len * seq_len + i * seq_len;
            float max_val = f16_to_f32(y_ptr[offset]);
            for (size_t j = 1; j < total_seq_len; j++) {
                max_val = std::max(max_val, f16_to_f32(y_ptr[offset + j]));
            }
            float sum = 0.;
            for (size_t j = 0; j < total_seq_len; j++) {
                float new_val = std::exp(f16_to_f32(y_ptr[offset + j]) - max_val);
                sum += new_val;
            }
            for (size_t j = 0; j < total_seq_len; j++) {
                float new_val = std::exp(f16_to_f32(y_ptr[offset + j]) - max_val) / sum;
                y_ptr[offset + j] = f32_to_f16(new_val);
            }
        }
    }
}
