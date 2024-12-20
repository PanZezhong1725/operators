#ifndef CONCAT_H
#define CONCAT_H

#include "../../export.h"
#include "../../operators.h"

// Concat描述符结构
typedef struct ConcatDescriptor {
    Device device;  // 设备类型（例如 DevCpu、DevNvGpu）
    uint64_t axis;  // 拼接轴（从0开始）
} ConcatDescriptor;

typedef ConcatDescriptor *infiniopConcatDescriptor_t;

// 创建Concat描述符
__C __export infiniopStatus_t infiniopCreateConcatDescriptor(infiniopHandle_t handle,
                                                             infiniopConcatDescriptor_t *desc_ptr,
                                                             infiniopTensorDescriptor_t y,
                                                             infiniopTensorDescriptor_t *x,
                                                             uint64_t num_inputs,
                                                             uint64_t axis);

// 执行Concat操作
__C __export infiniopStatus_t infiniopConcat(infiniopConcatDescriptor_t desc,
                                             void *y,
                                             void const **x,
                                             void *stream);

// 销毁Concat描述符
__C __export infiniopStatus_t infiniopDestroyConcatDescriptor(infiniopConcatDescriptor_t desc);

#endif
