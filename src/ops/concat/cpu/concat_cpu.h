#ifndef __CPU_CONCAT_H__
#define __CPU_CONCAT_H__
#include "operators.h"
#include <vector>
#include <cstring>

// 支持高维拼接的CPU-specific Concat描述符
struct ConcatCpuDescriptor {
    Device device;                                // 设备类型（例如 DevCpu）
    DT dtype;                                    // 数据类型
    uint64_t axis;                               // 拼接轴（从0开始）
    uint64_t ndim;                               // 张量维度
    uint64_t num_inputs;                         // 输入张量的数量
    std::vector<std::vector<uint64_t>> input_shapes;  // 输入张量的形状
    std::vector<std::vector<uint64_t>> input_strides;  // 输入张量的步长
    std::vector<uint64_t> output_shape;              // 输出张量的形状
    std::vector<uint64_t> output_stride;              // 输出张量的步长
};



typedef struct ConcatCpuDescriptor *ConcatCpuDescriptor_t;

// 创建Concat描述符
infiniopStatus_t cpuCreateConcatDescriptor(infiniopHandle_t handle,
                                           ConcatCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t y,
                                           infiniopTensorDescriptor_t *x,
                                           uint64_t num_inputs,
                                           uint64_t axis);

// 执行Concat操作
infiniopStatus_t cpuConcat(ConcatCpuDescriptor_t desc,
                           void *y,
                           void const **x,
                           void *stream);

// 销毁Concat描述符
infiniopStatus_t cpuDestroyConcatDescriptor(ConcatCpuDescriptor_t desc);

void linearToMultiDim(std::vector<uint64_t> &indices,
                      uint64_t linear_idx,
                      const std::vector<uint64_t> &shape,
                      uint64_t exclude_axis);

uint64_t computeOffset(const std::vector<uint64_t> &indices,
                       const std::vector<uint64_t> &stride);

#endif
