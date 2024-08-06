#ifndef __CNNL_MATMUL_H__
#define __CNNL_MATMUL_H__

#include "../blas.h"
#include "cnnl.h"
#include "cnnl_extra.h"
#include "operators.h"

struct MatmulBangDescriptor {
    Device device;
    MatmulBangDescriptor(Device device);
};

inline void setMatrixTensorEx(cnnlTensorDescriptor_t desc, const BlasMatrix &matrix, bool trans = false) {
    int ndim = matrix.ndim;
    int batch = matrix.batch;
    int stride = static_cast<int>(matrix.stride);
    int rows = matrix.rows;
    int cols = matrix.cols;
    int row_stride = matrix.row_stride;
    int col_stride = matrix.col_stride;

    if (ndim == 3) {
        std::vector<int> dim_size = {batch, rows, cols};
        std::vector<int> dim_stride = {stride, row_stride, col_stride};
        cnnlSetTensorDescriptorEx(desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
                                  dim_size.size(), dim_size.data(), dim_stride.data());
    } else if (ndim == 2) {
        std::vector<int> dim_size = {rows, cols};
        std::vector<int> dim_stride = {row_stride, col_stride};
        cnnlSetTensorDescriptorEx(desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
                                  dim_size.size(), dim_size.data(), dim_stride.data());
    }
}

void matmul_cnnl_f16(Tensor c, float beta, Tensor a, Tensor b, float alpha, void *stream);

#endif// __CNNL_MATMUL_H__
