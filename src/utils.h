#ifndef __UTILS_H__
#define __UTILS_H__

#include "data_type.h"
#include "tensor.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>

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

inline std::vector<int64_t> get_byte_strides(infiniopTensorDescriptor_t desc){
    int64_t dsize = desc->dt.size;
    std::vector<int64_t> strides(desc->ndim);
    for (uint64_t i = 0; i < desc->ndim; i++){
        strides[i] = dsize * desc->strides[i];
    }

    return strides;
}


#endif// __UTILS_H__
