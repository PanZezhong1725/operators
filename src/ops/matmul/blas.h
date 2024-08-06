#ifndef __BLAS_H__
#define __BLAS_H__

#include "../utils.h"
#include "operators.h"
#include <algorithm>
#include <stdint.h>

typedef struct BlasMatrix {
    int ndim;
    int batch;
    int64_t stride;
    int rows;
    int cols;
    int row_stride;
    int col_stride;

    BlasMatrix() {}

    BlasMatrix(TensorLayout *layout) {
        if (layout->ndim == 2) {
            this->ndim = 2;
            this->batch = 1;
            this->stride = 0;
            this->rows = layout->shape[0];
            this->cols = layout->shape[1];
            this->row_stride = layout->strides[0] / layout->dt.size;
            this->col_stride = layout->strides[1] / layout->dt.size;
        } else if (layout->ndim == 3) {
            this->ndim = 3;
            this->batch = layout->shape[0];
            this->stride = this->batch == 1 ? 0 : layout->strides[0] / layout->dt.size;
            this->rows = layout->shape[1];
            this->cols = layout->shape[2];
            this->row_stride = layout->strides[1] / layout->dt.size;
            this->col_stride = layout->strides[2] / layout->dt.size;
        } else {
            PANIC(InvalidMatrixShape);
        }

        if (this->row_stride != 1 && this->col_stride != 1) {
            ASSERT(false);
            PANIC(MatrixIsNotContiguous);
        }
    }

    bool match_batch(int batch) const {
        return this->batch == batch || this->batch == 1;
    }

    void transpose() {
        std::swap(rows, cols);
        std::swap(row_stride, col_stride);
    }

    int ld() const {
        if (this->row_stride == 1) {
            return this->col_stride;
        } else {
            return this->row_stride;
        }
    }
} BlasMatrix;

struct MatmulInfo {
    BlasMatrix a_matrix;
    BlasMatrix b_matrix;
    BlasMatrix c_matrix;

    void const *a_ptr;
    void const *b_ptr;
    void *c_ptr;

    int m, n, k, batch;

    MatmulInfo(Tensor c, Tensor a, Tensor b, bool col_major = true) {
        a_matrix = BlasMatrix(a.layout);
        b_matrix = BlasMatrix(b.layout);
        c_matrix = BlasMatrix(c.layout);

        a_ptr = a.data;
        b_ptr = b.data;
        c_ptr = c.data;

        ASSERT_EQ(c_matrix.rows, a_matrix.rows);// m
        ASSERT_EQ(c_matrix.cols, b_matrix.cols);// n
        ASSERT_EQ(a_matrix.cols, b_matrix.rows);// k

        batch = c_matrix.batch;
        if (!a_matrix.match_batch(batch) || !b_matrix.match_batch(batch)) {
            PANIC(InvalidBatchSize);
        }

        if ((col_major && c_matrix.col_stride == 1) || (!col_major && c_matrix.row_stride == 1)) {
            c_matrix.transpose();
            b_matrix.transpose();
            a_matrix.transpose();
            std::swap(a_matrix, b_matrix);
            std::swap(a_ptr, b_ptr);
        }

        m = c_matrix.rows;
        n = c_matrix.cols;
        k = a_matrix.cols;
    }
};

#endif// __BLAS_H__
