#ifndef __TRANSPOSE_COMMON_H__
#define __TRANSPOSE_COMMON_H__

#include "operators.h"
#include <algorithm>
#include <cstddef>

/**
 * @brief Enum type that specifies what mode the transpose operation should use
 */
enum TransposeMode {
    FULL_CONTIGUOUS_COPY,
    OUTPUT_CONTIGUOUS_COPY,
    NON_CONTIGUOUS_COPY,
};

// check if the shapes of the two tensors are the reverse of each other
inline bool are_reverse(const uint64_t *arr1, const uint64_t *arr2, uint64_t n) {
    for (uint64_t i = 0, j = n - 1; i < j; ++i, --j) {
        if (arr1[i] != arr2[j] || arr1[j] != arr2[i]) {
            return false;
        }
    }
    return true;
}

// check if the perm indicates no change in the shape
inline bool is_same(const uint64_t *perm, uint64_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (perm[i] != i) {
            return false;
        }
    }
    return true;
}

// check if a tensor can be squeezed to 1D
inline bool can_squeeze_to_1D(const uint64_t *shape, uint64_t ndim) {
    return std::count_if(shape, shape + ndim, [](uint64_t dim) { return dim > 1; }) == 1;
}

#endif