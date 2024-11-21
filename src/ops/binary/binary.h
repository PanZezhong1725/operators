#ifndef BINARY_H
#define BINARY_H

#include "export.h"
#include "operators.h"
#include <cstddef>

typedef struct BinaryDescriptor {
    Device device;
} BinaryDescriptor;

typedef BinaryDescriptor *infiniopBinaryDescriptor_t;

/**
 * @brief Represents all the currently defined binary operations.
 */
struct BinaryMode {
    /** 
     * Note: new operation type should be added **BEFORE** the "Count" type
     *       "Count" should remain as the last element. New Arithmetic operations should be
     *       added in the 'Arithmetic operations' section, i.e., before the 'Logical operations:' section.
     */
    enum Mode {
        // Arithmetic operations:
        Add,
        Subtract,
        Multiply,
        Divide,
        Pow,
        Mod,
        Max,
        Min,
        BitwiseAnd,
        BitwiseOr,
        BitwiseXor,

        // Logical operations:
        // **TODO Not currently supported**
        // Requires Boolean data type
        And,
        Or,
        Xor,
        Less,
        LessOrEqual,
        Equal,
        Greater,
        GreaterOrEqual,

        Count,///< Number of binary operation types (marker for counting purposes).
    };

    // This static constant holds the total number of defined binary operations.
    static const size_t numBinaryMode = Count;
};

__C infiniopStatus_t infiniopCreateBinaryDescriptor(infiniopHandle_t handle,
                                                    infiniopBinaryDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t c,
                                                    infiniopTensorDescriptor_t a,
                                                    infiniopTensorDescriptor_t b,
                                                    int mode);

__C infiniopStatus_t infiniopBinary(infiniopBinaryDescriptor_t desc,
                                    void *c,
                                    void const *a,
                                    void const *b,
                                    void *stream);

__C infiniopStatus_t infiniopDestroyBinaryDescriptor(infiniopBinaryDescriptor_t desc);


#endif
