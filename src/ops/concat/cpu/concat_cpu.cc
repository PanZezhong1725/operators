#include "concat_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateConcatDescriptor(
    infiniopHandle_t handle,
    ConcatCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t *x,
    uint64_t num_inputs,
    int64_t axis) {
    if (y == nullptr || x == nullptr || desc_ptr == nullptr || num_inputs == 0) {
        return STATUS_BAD_PARAM;
    }

    if (!is_contiguous(y)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    int64_t ndim = y->ndim;  
    if (axis >= ndim || axis < -ndim) {
        return STATUS_BAD_PARAM;
    }

    if(axis < 0){
        axis = axis + ndim;
    }

    uint64_t total_size = 0;  
    std::vector<std::vector<uint64_t>> input_shapes(num_inputs);  

    std::vector<uint64_t> output_shape(y->shape, y->shape + ndim);

    for (size_t i = 0; i < num_inputs; ++i) {

        if (!is_contiguous(x[i])) {
            return STATUS_BAD_TENSOR_STRIDES;
        }

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

        input_shapes[i] = std::vector<uint64_t>(x[i]->shape, x[i]->shape + ndim);
        total_size += x[i]->shape[axis];
    }

    if (total_size != y->shape[axis]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new ConcatCpuDescriptor{
        DevCpu,
        y->dt,
        axis,
        num_inputs,
        input_shapes,
        output_shape,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyConcatDescriptor(ConcatCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

template <typename T>
infiniopStatus_t concatCompute(const ConcatCpuDescriptor_t& desc,
                               T* y,
                               void const** x) {
    int64_t axis = desc->axis;
    uint64_t num_inputs = desc->num_inputs;
    const std::vector<std::vector<uint64_t>>& input_shapes = desc->input_shapes;
    const std::vector<uint64_t>& output_shape = desc->output_shape;

    size_t blockOffsetInner = 1;
    for (size_t i = output_shape.size() - 1; i > axis; --i) {
        blockOffsetInner *= output_shape[i];
    }
    size_t blockOffset = output_shape[axis] * blockOffsetInner;

    for (size_t i = 0; i < num_inputs; ++i) {
        const std::vector<uint64_t>& input_shape = input_shapes[i];

        size_t dimOffset = 0;
        for (size_t j = 0; j < i; ++j) {
            dimOffset += input_shapes[j][axis];
        }

        size_t localBlockOffset = 1;
        for (size_t j = input_shape.size() - 1; j >= axis && j != static_cast<size_t>(-1); --j) {
            localBlockOffset *= input_shape[j];
        }
        
        size_t innerOffset = blockOffsetInner * dimOffset;
        size_t inSize = 1;
        for (auto dim : input_shape) {
            inSize *= dim;
        }

        T* input_data = static_cast<T*>(const_cast<void*>(x[i]));

        #pragma omp parallel for
        for (size_t iOffset = 0; iOffset < inSize; ++iOffset) {

            size_t oOffset = iOffset % localBlockOffset + innerOffset +
                             iOffset / localBlockOffset * blockOffset;

            y[oOffset] = input_data[iOffset];
        }
    }

    return STATUS_SUCCESS; 
}

infiniopStatus_t cpuConcat(ConcatCpuDescriptor_t desc,
                           void *y,
                           void const **x,
                           void *stream) {
    if (desc->dtype == F16) {
        return concatCompute<uint16_t>(desc, reinterpret_cast<uint16_t*>(y), x);
    }
    if (desc->dtype == F32) {
        return concatCompute<float>(desc, reinterpret_cast<float*>(y), x);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
