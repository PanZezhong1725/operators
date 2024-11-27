#ifndef __CONV_BIAS_ACT_COMMON_H__
#define __CONV_BIAS_ACT_COMMON_H__

#include "operators.h"
#include <cstddef>

/**
 * @brief Specifies the type of activation function
 */
struct ActivationMode {

    enum Mode {
        // activation functions
        IDENTITY,
        RELU,
        SIGMOID,
        // Count
        // NOTE: new activation functions should add before "Count"
        Count,
    };
    constexpr static size_t numOfActivationFunctions = Mode::Count;
};

#endif