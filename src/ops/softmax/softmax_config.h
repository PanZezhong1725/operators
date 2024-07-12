#ifndef __SOFTMAX_CONFIG_H__
#define __SOFTMAX_CONFIG_H__

typedef struct SoftmaxCudaConfig {
    // The upper bound of softmax dimension (axis)
    unsigned int max_dim;
} SoftmaxCudaConfig;

#endif // __SOFTMAX_CONFIG_H__
