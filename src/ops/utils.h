#ifndef __UTILS_H__
#define __UTILS_H__

#include "data_type.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>

/* This file contains some useful macros and helper functions */

// check if an expression is true, and if not, print an error message and abort the program
inline void assert_true(int expr, const char *msg, const char *file, int line) {
    if (!expr) {
        fprintf(stderr, "\033[31mAssertion failed:\033[0m %s at file %s, line %d\n", msg, file, line);
        exit(EXIT_FAILURE);
    }
}

#define ASSERT(expr) assert_true(expr, #expr " is false", __FILE__, __LINE__)
#define ASSERT_EQ(a, b) assert_true((a) == (b), #a " != " #b, __FILE__, __LINE__)
#define ASSERT_VALID_PTR(a) assert_true((a) != nullptr, #a " is nullptr", __FILE__, __LINE__)

#define PANIC(EXPR)                                             \
    printf("Error at %s:%d - %s\n", __FILE__, __LINE__, #EXPR); \
    exit(EXIT_FAILURE)

#define ROUND_UP_DIV(x, y) ((x + y - 1) / y)

#define CHECK_ERROR(call, target, errCode)            \
    do {                                              \
        if (auto value = (call); value == (target)) { \
            return (errCode);                         \
        }                                             \
    } while (0)
#define CREATE_CHECK_ERROR(expr, value, target, errCode) \
    expr;                                                \
    CHECK_ERROR(value, target, errCode)

// check if two data layouts (types) are equal
inline bool dtype_eq(DataLayout a, DataLayout b) {
    union TypePun {
        DataLayout layout;
        int i;
    } pun;
    pun.layout = a;
    auto a_ = pun.i;
    pun.layout = b;
    auto b_ = pun.i;
    return a_ == b_;
}

inline std::vector<int64_t> get_byte_strides(infiniopTensorDescriptor_t desc) {
    int64_t dsize = desc->dt.size;
    std::vector<int64_t> strides(desc->ndim);
    for (uint64_t i = 0; i < desc->ndim; i++) {
        strides[i] = dsize * desc->strides[i];
    }

    return strides;
}

inline bool is_contiguous(const uint64_t *shape, const int64_t *strides, uint64_t n) {
    for (int64_t expected_stride = 1, i = n - 1; i > 0; --i) {
        if (strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= shape[i];
    }
    return true;
}

inline bool is_contiguous(const infiniopTensorDescriptor_t &desc) {
    return is_contiguous(desc->shape, desc->strides, desc->ndim);
}

// calculate the broadcasted shape for two tensors
inline bool getBroadcastShape(const uint64_t *shape1, uint64_t ndim1,
                              const uint64_t *shape2, uint64_t ndim2,
                              uint64_t *broadcast_shape, uint64_t *padded_shape1,
                              uint64_t *padded_shape2, uint64_t max_rank) {
    // prepending and initializing
    std::fill(padded_shape1, padded_shape1 + max_rank, 1);
    std::fill(padded_shape2, padded_shape2 + max_rank, 1);
    std::copy(shape1, shape1 + ndim1, padded_shape1 + max_rank - ndim1);
    std::copy(shape2, shape2 + ndim2, padded_shape2 + max_rank - ndim2);

    // compute broadcasted shape
    for (int i = 0; i < max_rank; ++i) {
        if (padded_shape1[i] == padded_shape2[i] || padded_shape1[i] == 1 || padded_shape2[i] == 1) {
            broadcast_shape[i] = std::max(padded_shape1[i], padded_shape2[i]);
        } else {
            return false;
        }
    }

    return true;
}

// check if the shape of tensor c is valid after broadcasting tensors a and b and also get the broadcasted shapes
inline bool isValidBroadcastShape(infiniopTensorDescriptor_t a, infiniopTensorDescriptor_t b, infiniopTensorDescriptor_t c,
                                  uint64_t *broadcast_shape, uint64_t *padded_shape1, uint64_t *padded_shape2, uint64_t broadcast_ndim) {
    if (broadcast_ndim != c->ndim || !getBroadcastShape(a->shape, a->ndim, b->shape, b->ndim, broadcast_shape, padded_shape1, padded_shape2, broadcast_ndim)) {
        return false;
    }
    return std::equal(broadcast_shape, broadcast_shape + broadcast_ndim, c->shape);
}

// check if the shape of tensor c is valid after broadcasting tensors a and b
inline bool isValidBroadcastShape(infiniopTensorDescriptor_t a, infiniopTensorDescriptor_t b, infiniopTensorDescriptor_t c) {
    uint64_t broadcast_ndim = std::max(a->ndim, b->ndim);
    uint64_t broadcast_shape[broadcast_ndim];
    uint64_t padded_shape1[broadcast_ndim];
    uint64_t padded_shape2[broadcast_ndim];
    return isValidBroadcastShape(a, b, c, broadcast_shape, padded_shape1, padded_shape2, broadcast_ndim);
}


inline uint64_t get_byte_size(infiniopTensorDescriptor_t desc) {
    uint64_t dsize = desc->dt.size;
    uint64_t size = 1;
    for (uint64_t i = 0; i < desc->ndim; i++) {
        size *= desc->shape[i];
    }
    return size * dsize;
}

#endif// __UTILS_H__
