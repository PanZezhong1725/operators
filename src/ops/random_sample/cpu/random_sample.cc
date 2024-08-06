#include "random_sample_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <cmath>



void random_sample_cpu_f16(Tensor source, Tensor indices, Tensor index, float random, float topp, int topk) {
    ASSERT_EQ(source.layout->ndim, 1);
    ASSERT_EQ(indices.layout->ndim, 1);
    auto voc = source.layout->shape[0];
    auto source_ = reinterpret_cast<uint16_t const *>(source.data);
    auto indices_ = reinterpret_cast<int const *>(indeces.data);
    auto index_ = reinterpret_cast<int const *>(index.data);
    float p = random * min(topp * f16_to_f32(source_[voc - 1]), f16_to_f32(source_[topk - 1]));

    for (int i = 0; i < voc; ++i) {
        if(f16_to_f32(source_[i]) >= p){
            index_[0] = indices_[i];
            break;
        }
        
    }
}
