#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include "random_sample_cpu.h"
#include <cmath>


infiniopStatus_t cpuCreateRandomSampleDescriptor(infiniopHandle_t,
                                                 RandomSampleCpuDescriptor_t *desc_ptr,
                                                 infiniopTensorDescriptor_t probs) {
    int ndim = probs->ndim;
    if (ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!dtype_eq(probs->dt, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    int voc = probs->shape[0];

    *desc_ptr = new RandomSampleCpuDescriptor{
        DevCpu,
        probs->dt,
        voc};

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGetRandomSampleWorkspaceSize(RandomSampleCpuDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyRandomSampleDescriptor(RandomSampleCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}


void causal_softmax_cpu_f16(RandomSampleCpuDescriptor_t desc,
                            void *result,
                            void *probs,
                            float random_val,
                            float topp,
                            int topk,
                            float temperature) {
    int voc = desc->voc;
    auto logits_ = reinterpret_cast<uint16_t *>(probs);
    auto index_ = reinterpret_cast<uint64_t *>(result);

    // 如果k大于voc，调整k为voc
    if (topk > voc) {
        topk = voc;
    }
    //排序得到前k个最大值，按照从大到小顺序存储在logits_前k个位置里面
    uint64_t *indexTmp = (uint64_t *) malloc(voc * sizeof(uint64_t));
    for (int i = 0; i < voc; i++) {
        indexTmp[i] = i;
    }
    for (int i = 0; i < topk; i++) {
        for (int j = i + 1; j < voc; j++) {
            if (f16_to_f32(logits_[i]) < f16_to_f32(logits_[j])) {
                float M = f16_to_f32(logits_[i]);
                logits_[i] = logits_[j];
                logits_[j] = f32_to_f16(M);


                int index = indexTmp[i];
                indexTmp[i] = indexTmp[j];
                indexTmp[j] = index;
            }
        }
    }

    //做类似于softmax的temperature变换
    float reduceM = f16_to_f32(logits_[0]);
    float reduceS = 0.0f;
    for (int i = 0; i < voc; i++) {
        reduceS += std::exp((f16_to_f32(logits_[i]) - reduceM) / temperature);
    }
    for (int i = 0; i < voc; i++) {
        logits_[i] = f32_to_f16(std::exp((f16_to_f32(logits_[i]) - reduceM) / temperature) / reduceS);
    }
    //在前k个元素里面利用topp选取不超过topp的元素作为数据集
    float tmp = 0.0f;
    int end = 0;
    for (end = 0; end < topk; end++) {
        tmp += f16_to_f32(logits_[end]);
        if (tmp >= topp) {
            break;
        }
    }
    //printf("%d\n", end);
    if (end < topk - 1) {
        end += 1;
    } else {
        end = topk;
    }
    //利用随机数随机输出满足同时满足topk,topp的某个元素在原始向量的index

    float sum_s = 0.0f;
    for (int i = 0; i < end; i++) {
        sum_s += f16_to_f32(logits_[i]);
    }
    random_val *= sum_s;

    sum_s = 0.0f;
    for (int i = 0; i < end; i++) {
        sum_s += f16_to_f32(logits_[i]);
        if (random_val < sum_s) {
            index_[0] = indexTmp[i];
            break;
        }
    }
    free(indexTmp);
}

infiniopStatus_t cpuRandomSample(RandomSampleCpuDescriptor_t desc,
                                 void *workspace,
                                 uint64_t workspace_size,
                                 void *result,
                                 void *probs,
                                 float random_val,
                                 float topp,
                                 int topk,
                                 float temperature,
                                 void *stream) {
    if (dtype_eq(desc->dtype, F16)) {
        causal_softmax_cpu_f16(desc,
                               result,
                               probs,
                               random_val,
                               topp,
                               topk,
                               temperature);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
