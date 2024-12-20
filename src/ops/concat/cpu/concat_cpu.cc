#include "concat_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"


infiniopStatus_t cpuCreateConcatDescriptor(
    infiniopHandle_t handle,
    ConcatCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t *x,
    uint64_t num_inputs,
    uint64_t axis) {
    if (y == nullptr || x == nullptr || desc_ptr == nullptr || num_inputs == 0) {
        return STATUS_BAD_PARAM;
    }

    uint64_t ndim = y->ndim;  // 输出张量维度
    if (axis >= ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    uint64_t total_size = 0;  // 拼接轴的总大小
    std::vector<std::vector<uint64_t>> input_shapes(num_inputs);  // 输入张量形状
    std::vector<std::vector<uint64_t>> input_strides(num_inputs);  // 输入张量步长

    // 提取输出张量的形状和步长
    std::vector<uint64_t> output_shape(y->shape, y->shape + ndim);
    std::vector<uint64_t> output_stride(y->strides, y->strides + ndim);

    // 验证输入张量的形状和步长，并记录形状信息
    for (size_t i = 0; i < num_inputs; ++i) {

        if (x[i]->dt != y->dt) {
            return STATUS_BAD_TENSOR_DTYPE;
        }

        if (x[i]->ndim != ndim) {
            return STATUS_BAD_TENSOR_SHAPE;
        }

        for (size_t j = 0; j < ndim; ++j) {
            if (j != axis && x[i]->shape[j] != y->shape[j]) {
                return STATUS_BAD_TENSOR_SHAPE;
            }
        }
        // 记录每个输入张量的形状和步长
        input_shapes[i] = std::vector<uint64_t>(x[i]->shape, x[i]->shape + ndim);
        input_strides[i] = std::vector<uint64_t>(x[i]->strides, x[i]->strides + ndim);

        // 累加拼接轴的总大小
        total_size += x[i]->shape[axis];
    }

    // 验证输出张量形状是否匹配
    if (total_size != y->shape[axis]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    // 初始化Concat描述符
    *desc_ptr = new ConcatCpuDescriptor{
        DevCpu,
        y->dt,
        axis,
        ndim,
        num_inputs,
        input_shapes,
        input_strides,
        output_shape,
        output_stride
    };

    return STATUS_SUCCESS;
}


// 销毁Concat描述符
infiniopStatus_t cpuDestroyConcatDescriptor(ConcatCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}


// Helper function to handle different data types
template <typename T>
infiniopStatus_t concatCompute(const ConcatCpuDescriptor_t& desc,
                               T* y,
                               void const** x) {
    uint64_t axis = desc->axis;
    uint64_t num_inputs = desc->num_inputs;
    const std::vector<std::vector<uint64_t>> &input_shapes = desc->input_shapes;
    const std::vector<std::vector<uint64_t>> &input_strides = desc->input_strides;
    const std::vector<uint64_t> &output_shape = desc->output_shape;
    const std::vector<uint64_t> &output_stride = desc->output_stride;
    uint64_t ndim = desc->ndim;


    // 计算拼接轴之前的总元素数（外层维度）
    uint64_t outer_dim = 1;
    for (uint64_t d = 0; d < axis; ++d) {
        outer_dim *= output_shape[d];
    }

    // 计算拼接轴之后的总元素数（内层维度）
    uint64_t inner_dim = 1;
    for (uint64_t d = axis + 1; d < ndim; ++d) {
        inner_dim *= output_shape[d];
    }

    // 计算每个输入张量在拼接轴上的偏移量
    std::vector<uint64_t> dim_offsets(num_inputs, 0);
    for (uint64_t i = 1; i < num_inputs; ++i) {
        dim_offsets[i] = dim_offsets[i - 1] + input_shapes[i - 1][axis];
    }

    // 并行化外部循环
    #pragma omp parallel for
    for (uint64_t od = 0; od < outer_dim; ++od) {
        // 计算当前外层索引在各维度上的位置
        // 例如，如果外层维度为 [d0, d1, ..., d(axis-1)]
        // 则 od 可以被分解为 d0 * (output_stride[0]) + d1 * (output_stride[1]) + ...
        std::vector<uint64_t> indices(ndim, 0);
        uint64_t tmp = od;
        for (uint64_t d = 0; d < axis; ++d) {
            indices[d] = tmp / output_stride[d];
            tmp %= output_stride[d];
        }

        for (uint64_t i = 0; i < num_inputs; ++i) {
            // 输入张量的拼接轴上的偏移
            uint64_t input_axis_offset = dim_offsets[i];

            // 遍历拼接轴上的所有元素
            for (uint64_t a = 0; a < input_shapes[i][axis]; ++a) {
                // 设置当前拼接轴的索引
                indices[axis] = a + input_axis_offset;

                // 计算输出张量的线性索引
                uint64_t y_offset = 0;
                for (uint64_t d = 0; d < ndim; ++d) {
                    y_offset += indices[d] * output_stride[d];
                }

                // 计算输入张量的线性索引
                uint64_t x_offset = 0;
                for (uint64_t d = 0; d < ndim; ++d) {
                    x_offset += indices[d] * input_strides[i][d];
                }

                // 复制数据
                y[y_offset] = reinterpret_cast<const T*>(x[i])[x_offset];
            }
        }
    }

    return STATUS_SUCCESS;
}

// 主拼接函数
infiniopStatus_t cpuConcat(ConcatCpuDescriptor_t desc,
                           void *y,
                           void const **x,
                           void *stream) {
    // 根据数据类型调用相应的模板实例
    switch (desc->dtype.size) {
        case sizeof(float): // FLOAT32
            return concatCompute<float>(desc, reinterpret_cast<float*>(y), x);
        // 可以根据需要添加更多数据类型
        default:
            return STATUS_SUCCESS;
    }
}





// infiniopStatus_t cpuConcat(ConcatCpuDescriptor_t desc,
//                            void *y,
//                            void const **x,
//                            void *stream) {
//     // 从描述符中获取必要信息
//     uint64_t axis = desc->axis;  // 拼接轴
//     uint64_t num_inputs = desc->num_inputs;  // 输入张量数量
//     const std::vector<std::vector<uint64_t>> &input_shapes = desc->input_shapes;  // 输入张量形状
//     const std::vector<std::vector<uint64_t>> &input_strides = desc->input_strides;  // 输入张量步长
//     const std::vector<uint64_t> &output_shape = desc->output_shape;  // 输出张量形状
//     const std::vector<uint64_t> &output_stride = desc->output_stride;  // 输出张量步长

//     DT dtype = desc->dtype;
//     size_t element_size = dtype.size;
//     uint64_t ndim = desc->ndim;

//     // 初始化累计偏移量，用于拼接轴的起始位置
//     uint64_t cumulative_axis_offset = 0;

//     // 遍历每个输入张量
//     for (uint64_t tensor_idx = 0; tensor_idx < num_inputs; ++tensor_idx) {
//         const uint8_t *x_data = static_cast<const uint8_t *>(x[tensor_idx]);
//         const auto &x_shape = input_shapes[tensor_idx];
//         const auto &x_stride = input_strides[tensor_idx];

//         uint64_t axis_size = x_shape[axis];  // 当前张量在拼接轴上的大小

//         // 计算非拼接轴的遍历总数（用于确定每个块的偏移量）
//         uint64_t outer_loops = 1;
//         for (uint64_t i = 0; i < ndim; ++i) {
//             if (i != axis) {
//                 outer_loops *= x_shape[i];
//             }
//         }

//         uint64_t x_size=1;
//         for(int i=0; i < ndim; i++){
//             x_size = x_size * x_shape[i];
//         }
//         x_size = x_size * element_size;

//         // 遍历非拼接轴的所有元素块
//         for (uint64_t outer_idx = 0; outer_idx < outer_loops; ++outer_idx) {
//             // 将线性索引转换为多维索引
//             std::vector<uint64_t> indices(ndim, 0);
//             linearToMultiDim(indices, outer_idx, x_shape, axis);

//             // 计算输入和输出张量的偏移量
//             uint64_t input_block_offset =computeOffset(indices, x_stride) * element_size;
//             indices[axis] += cumulative_axis_offset;
//             uint64_t output_block_offset =computeOffset(indices, output_stride) * element_size;

//             // 计算剩余空间
//             // uint64_t remaining_space_in_x = x_size - input_block_offset;
           
//             uint64_t remaining_space_in_x = (x_stride[axis] - input_block_offset % x_stride[axis]) * element_size;

//             printf("remaining_space_in_x:  %llu\n",remaining_space_in_x);

//             // 计算当前块的大小，但不能超过剩余空间
//             uint64_t block_size = std::min(axis_size * x_stride[axis] * element_size, remaining_space_in_x);


//             memcpy(static_cast<uint8_t *>(y) + output_block_offset, x_data + input_block_offset, block_size);
//         }

//         // 更新累计偏移量
//         cumulative_axis_offset += axis_size;
//     }

//     return STATUS_SUCCESS;

// }


void linearToMultiDim(std::vector<uint64_t> &indices,
                      uint64_t linear_idx,
                      const std::vector<uint64_t> &shape,
                      uint64_t exclude_axis) {
    uint64_t ndim = shape.size();
    for (int64_t dim = ndim - 1; dim >= 0; --dim) {
        if (dim == exclude_axis) 
            continue;  // 跳过拼接轴
        indices[dim] = linear_idx % shape[dim];
        linear_idx /= shape[dim];
    }
}


uint64_t computeOffset(const std::vector<uint64_t> &indices,
                       const std::vector<uint64_t> &stride) {
    uint64_t offset = 0;
    for (uint64_t dim = 0; dim < indices.size(); ++dim) {
        offset += indices[dim] * stride[dim];
    }
    return offset;
}



// uint64_t remaining_space_in_x = (x_stride[axis] - input_block_offset % x_stride[axis]) * element_size;

// printf("remaining_space_in_x:  %llu\n",remaining_space_in_x);

// // 每次拷贝当前块
// uint64_t block_size = axis_size * x_stride[axis] * element_size;