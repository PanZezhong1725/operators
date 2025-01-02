#include "concat.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateConcatDescriptor(CudaHandle_t handle,
                                            ConcatCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t *x,
                                            uint64_t num_inputs,
                                            int64_t axis){
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

    *desc_ptr = new ConcatCudaDescriptor{
        DevNvGpu,
        y->dt,
        axis,
        num_inputs,
        input_shapes,
        output_shape,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyConcatDescriptor(ConcatCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}