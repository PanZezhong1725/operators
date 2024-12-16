#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include "random_sample_cpu.h"
#include <cmath>


infiniopStatus_t cpuCreateRandomSampleDescriptor(infiniopHandle_t,
                                                 RandomSampleCpuDescriptor_t *desc_ptr, infiniopTensorDescriptor_t result,
                                                 infiniopTensorDescriptor_t probs) {
    int ndim = probs->ndim;
    if (ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!dtype_eq(probs->dt, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!dtype_eq(result->dt, U64))
        return STATUS_BAD_TENSOR_DTYPE;
    int voc = probs->shape[0];
    int rLength = result->shape[0];
    if (result->ndim != 1 && rLength != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    *desc_ptr = new RandomSampleCpuDescriptor{
        DevCpu,
        probs->dt,
        voc,
        result->dt,
        rLength};

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGetRandomSampleWorkspaceSize(RandomSampleCpuDescriptor_t desc, uint64_t *size) {
    *size = desc->voc * (sizeof(uint64_t) + sizeof(desc->dtype));
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyRandomSampleDescriptor(RandomSampleCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}


void random_sample_cpu_f16(RandomSampleCpuDescriptor_t desc,
                           void *workspace,
                           void *result,
                           void const *probs,
                           float random_val,
                           float topp,
                           int topk,
                           float temperature) {
    int voc = desc->voc;
    char *origin = reinterpret_cast<char *>(workspace);
    //排序得到前k个最大值，按照从大到小顺序存储在logits_前k个位置里面
    char *logitsTmp = origin + voc * sizeof(uint64_t);
    uint64_t *indexTmp = (uint64_t *) origin;
    uint16_t *logits_ = (uint16_t *) logitsTmp;


    auto source = reinterpret_cast<const uint16_t *>(probs);

    std::copy(source, source + voc, logits_);
    auto index_ = reinterpret_cast<uint64_t *>(result);

    // 如果k大于voc，调整k为voc
    if (topk > voc) {
        topk = voc;
    }

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
}
void random_sample_cpu_f16(RandomSampleCpuDescriptor_t desc,
                           void *workspace,
                           void *result,
                           void const *probs) {
    int voc = desc->voc;
    auto index_ = reinterpret_cast<uint64_t *>(result);
    auto source = reinterpret_cast<const uint16_t *>(probs);

    char *origin = reinterpret_cast<char *>(workspace);
    uint16_t *logits_ = (uint16_t *) origin;

    std::copy(source, source + voc, logits_);

    float M = f16_to_f32(logits_[0]);
    int index = 0;
    for (int j = 1; j < voc; j++) {
        if (M < f16_to_f32(logits_[j])) {
            M = f16_to_f32(logits_[j]);
            index = j;
        }
    }

    index_[0] = index;
}

infiniopStatus_t cpuRandomSample(RandomSampleCpuDescriptor_t desc,
                                 void *workspace,
                                 uint64_t workspace_size,
                                 void *result,
                                 void const *probs,
                                 float random_val,
                                 float topp,
                                 int topk,
                                 float temperature,
                                 void *stream) {
    if (dtype_eq(desc->dtype, F16)) {
        if (topp > 0 && topk > 1) {
            random_sample_cpu_f16(desc,
                                  workspace,
                                  result,
                                  probs,
                                  random_val,
                                  topp,
                                  topk,
                                  temperature);
        } else {
            random_sample_cpu_f16(desc,
                                  workspace,
                                  result,
                                  probs);
        }
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
