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
    int *indexTmp = (int *)malloc(voc * sizeof(int));
    for(int i = 0; i < voc; i++){
        indexTmp[i] = i;
    }
    for(int i = 0; i < topk; i++){
        for(int j = i + 1; j < voc; j++){
            if(f16_to_f32(logits_[i]) < f16_to_f32(logits_[j])){
                float M = f16_to_f32(logits_[i]);
                logits_[i] = logits_[j];
                logits_[j] = f32_to_f16(M);


                int index = indexTmp[i];
                indexTmp[i] = indexTmp[j];
                indexTmp[j] = index;
            }
        }
    }
    // for(int i = 0; i < topk; i++){
    //     printf("%d ", indexTmp[i]);
    // }
    // printf("\n");
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
    //printf("%d\n", end);
    if(end < topk - 1){
        end += 1;
    }
    else{
        end = topk;
    }
    //利用随机数随机输出满足同时满足topk,topp的某个元素在原始向量的index
    //float randomVal = (float)rand() / RAND_MAX;  
    float randomVal = 0.75;
    float sum_s = 0.0f;
    for(int i = 0; i < end; i++){
        sum_s += f16_to_f32(logits_[i]);
    }
    randomVal *= sum_s;
    //printf("%.5f\n", randomVal);
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
