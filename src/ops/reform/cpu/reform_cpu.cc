#include "reform_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <cstring>
#include <numeric>

inline int indices(int i, int ndim, int64_t *strides, uint64_t *shape) {
    int ans = 0;
    for (int j = ndim - 2; j >= 0; --j) {
        ans += (i % shape[j]) * strides[j];
        i /= shape[j];
    }
    return ans;
}

void copy_contiguous(uint8_t *dst_ptr, uint8_t const *src_ptr, int n, Tensor y, Tensor x) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        auto dst_offset = indices(i, y.layout->ndim, y.layout->strides, y.layout->shape);
        auto src_offset = indices(i, y.layout->ndim, x.layout->strides, x.layout->shape);
        std::memcpy(dst_ptr + dst_offset, src_ptr + src_offset, y.layout->shape[y.layout->ndim - 1] * y.layout->dt.size);
    }
}

union DataLayout_ {
    DataLayout i;
    unsigned short u;
};

void reform_cpu(Tensor y, Tensor x) {
    DataLayout_ dl_y, dl_x;
    dl_y.i = y.layout->dt;
    dl_x.i = x.layout->dt;
    ASSERT_EQ(dl_y.u, dl_x.u);
    ASSERT_EQ(y.layout->ndim, x.layout->ndim);
    auto ndim = y.layout->ndim;
    ASSERT(ndim >= 2);
    for (int i = 0; i < ndim; ++i) {
        ASSERT_EQ(y.layout->shape[i], x.layout->shape[i]);
    }
    ASSERT_EQ(y.layout->strides[ndim - 1], y.layout->dt.size);
    ASSERT_EQ(x.layout->strides[ndim - 1], x.layout->dt.size);
    unsigned int r = 0;
    if (ndim == 2) {
        r = y.layout->shape[0];
    } else if (ndim == 3) {
        r = y.layout->shape[0] * y.layout->shape[1];
    } else {
        for (int i = ndim - 3; i >= 1; --i) {
            ASSERT_EQ(y.layout->shape[i] * y.layout->strides[i], y.layout->strides[i - 1]);
            ASSERT_EQ(x.layout->shape[i] * x.layout->strides[i], x.layout->strides[i - 1]);
        }
        r = std::accumulate(y.layout->shape, y.layout->shape + ndim - 1, 1, std::multiplies<unsigned int>());
    }
    auto dst_ptr = reinterpret_cast<uint8_t *>(y.data);
    auto src_ptr = reinterpret_cast<uint8_t const *>(x.data);

    copy_contiguous(dst_ptr, src_ptr, r, y, x);
}
