#ifndef __CPU_CONCAT_H__
#define __CPU_CONCAT_H__
#include "operators.h"
#include <vector>
#include <cstring>

struct ConcatCpuDescriptor {
    Device device;                                
    DT dtype;                                    
    int64_t axis;                               
    uint64_t num_inputs;                        
    std::vector<std::vector<uint64_t>> input_shapes;  
    std::vector<uint64_t> output_shape;              
};

typedef struct ConcatCpuDescriptor *ConcatCpuDescriptor_t;

infiniopStatus_t cpuCreateConcatDescriptor(infiniopHandle_t handle,
                                           ConcatCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t y,
                                           infiniopTensorDescriptor_t *x,
                                           uint64_t num_inputs,
                                           int64_t axis);

infiniopStatus_t cpuConcat(ConcatCpuDescriptor_t desc,
                           void *y,
                           void const **x,
                           void *stream);

infiniopStatus_t cpuDestroyConcatDescriptor(ConcatCpuDescriptor_t desc);

#endif
