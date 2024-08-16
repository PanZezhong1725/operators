#include "random_sample_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <cmath>



void random_sample_cpu_f16(Tensor source, Tensor indices, float topp, int topk, float temperature) {
    ASSERT_EQ(source.layout->ndim, 1);
    ASSERT_EQ(indices.layout->ndim, 1);
    auto voc = source.layout->shape[0];
    auto logits_ = reinterpret_cast<uint16_t *>(source.data);
    auto index_ = reinterpret_cast<int *>(indices.data);
    
    // 如果k大于voc，调整k为voc  
    if (topk > voc) {  
        topk = voc;  
    }  
    //排序得到前k个最大值，按照从大到小顺序存储在logits_前k个位置里面
    for(int i = 0; i < voc; i++){
        printf("%.2f ", f16_to_f32(logits_[i]));
    }
    printf("\n");
    int *indexTmp = (int *)malloc(topk * sizeof(int));
    for(int i = 0; i < topk; i++){
        float M = f16_to_f32(logits_[i]);
        int index = i;
        for(int j = i + 1; j < voc; j++){
            if(M < f16_to_f32(logits_[j])){
                M = f16_to_f32(logits_[j]);
                index = j;
            }
        }
        float tmp = f16_to_f32(logits_[i]);
        logits_[i] = f32_to_f16(M);
        logits_[index] = f32_to_f16(tmp);
        indexTmp[i] = index;
        //printf("%.2f, %.2f, %d\n", M, tmp, indexTmp[i]);
    }

    //做类似于softmax的temperature变换
    float reduceM = f16_to_f32(logits_[0]);
    float reduceS = 0.0f;
    for(int i = 0; i < voc; i++){
        reduceS += std::exp((f16_to_f32(logits_[i]) - reduceM) / temperature);
    }
    for(int i = 0; i < voc; i++){
        logits_[i] = f32_to_f16(std::exp((f16_to_f32(logits_[i]) - reduceM) / temperature) / reduceS);
    }
    //在前k个元素里面利用topp选取不超过topp的元素作为数据集
    float tmp = 0.0f;
    int end = 0;
    for(end = 0; end < topk; end++){
        tmp += f16_to_f32(logits_[end]);
        if(tmp >= topp){
            break;
        }
    }
    //利用随机数随机输出满足同时满足topk,topp的某个元素在原始向量的index
    //float randomVal = (float)rand() / RAND_MAX;  
    float randomVal = 0.75;
    float sum_s = 0.0f;
    for(int i = 0; i < end; i++){
        sum_s += f16_to_f32(logits_[i]);
    }
    randomVal *= sum_s;
    sum_s = 0.0f;
    for(int i = 0; i < end; i++){
        sum_s += f16_to_f32(logits_[i]);
        if(randomVal < sum_s){
            index_[0] = indexTmp[i];
            break;
        }
    }
    free(indexTmp);
}
