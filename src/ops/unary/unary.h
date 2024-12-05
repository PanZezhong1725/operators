#ifndef UNARY_H
#define UNARY_H

#include "export.h"
#include "operators.h"
#include <cfenv>
#include <cstddef>

#define TAN_THRESHOLD 15000

typedef struct UnaryDescriptor {
    Device device;
} UnaryDescriptor;

typedef UnaryDescriptor *infiniopUnaryDescriptor_t;

/**
 * @brief Represents all the currently defined unary operations.
 */
struct UnaryMode {
    /** 
     * Note: new operation type should be added **BEFORE** the "Count" type
     *       "Count" should remain as the last element. New Arithmetic operations should be
     *       added in the 'Arithmetic operations' section, i.e., before the 'Logical operations:' section.
     */
    enum Mode {
        // Math operations:
        Abs,
        Exp,
        Log,
        Reciprocal,
        Sqrt,
        Neg,
        Ceil,
        Floor,
        Round,
        Sin,
        Cos,
        Tan,
        Asin,
        Acos,
        Atan,
        Sinh,
        Cosh,
        Tanh,
        Asinh,
        Acosh,
        Atanh,
        Relu,
        Sigmoid,
        Sign,
        Erf,

        // Bitwise operations:
        BitwiseNot,

        // Logical operations:
        // **TODO Not currently supported**
        // Requires Boolean data type
        Not,

        Count,///< Number of unary operation types (marker for counting purposes).
    };

    // This static constant holds the total number of defined unary operations.
    static const size_t numUnaryMode = Count;
};

__C infiniopStatus_t infiniopCreateUnaryDescriptor(infiniopHandle_t handle,
                                                   infiniopUnaryDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y,
                                                   infiniopTensorDescriptor_t x,
                                                   int mode);

__C infiniopStatus_t infiniopUnary(infiniopUnaryDescriptor_t desc,
                                   void *y,
                                   void const *x,
                                   void *stream);

__C infiniopStatus_t infiniopDestroyUnaryDescriptor(infiniopUnaryDescriptor_t desc);


#endif
